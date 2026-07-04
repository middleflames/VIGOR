import torch.nn as nn
import torch
from models import XVLMBase, load_pretrained
from models.deformable1d import DeformableAttention1D


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim),
    )


class Recursive_layer1(nn.Module):
    # this is for verb and noun
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_tag1 = nn.Linear(768, 4)
        self.bbox_head = build_mlp(input_dim=768, output_dim=4)
        self.tag_loss = nn.CrossEntropyLoss()

    def forward(self, embds, tag_labels):
        pos_tag1 = self.pos_tag1(embds)
        output_cls = embds[:, 1, :]
        output_coord = self.bbox_head(output_cls).sigmoid()
        tag_loss = self.tag_loss(pos_tag1.view(-1, 4), tag_labels.view(-1))
        return tag_loss, output_coord, output_cls


class Recursive_layer2(nn.Module):
    # this is for other context tags
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_tag = nn.Linear(768, 5)
        self.bbox_head = build_mlp(input_dim=768, output_dim=4)
        self.tag_loss = nn.CrossEntropyLoss()

    def forward(self, embds, tag_labels):
        pos_tag1 = self.pos_tag(embds)
        output_cls = embds[:, 2, :]
        output_coord = self.bbox_head(output_cls).sigmoid()
        tag_loss = self.tag_loss(pos_tag1.view(-1, 5), tag_labels.view(-1))
        return tag_loss, output_coord, output_cls


class XVLM_POSLIDAR(XVLMBase):
    def __init__(self, config):
        super().__init__(
            config,
            load_vision_params=False,
            load_text_params=False,
            use_contrastive_loss=False,
            use_matching_loss=False,
            use_mlm_loss=False,
            use_bbox_loss=True,
        )
        self.init_params = []
        self.recursive_layer1 = Recursive_layer1(config)
        self.recursive_layer2 = Recursive_layer2(config)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lidar_fc = nn.Linear(384, 768 * 32)
        self.deformable1 = DeformableAttention1D(dim=32)
        self.deformable2 = DeformableAttention1D(dim=32)

    def load_pretrained(
        self, ckpt_rpath, config, load_bbox_pretrain=False, is_eval=False
    ):
        print("### load_bbox_pretrain, ", load_bbox_pretrain)
        state_dict = load_pretrained(
            ckpt_rpath, config, is_eval=is_eval, load_text=True
        )
        msg = self.load_state_dict(state_dict, strict=False)
        print("load checkpoint from %s" % ckpt_rpath)
        print(
            "missing_keys: ", [p for p in msg.missing_keys if "vision_encoder" not in p]
        )
        print("unexpected_keys: ", msg.unexpected_keys)

    def predict_final_bbox(self, cls0, cls1, cls2):
        """
        Args:
            image_embeds: encoding full images

        Returns:
            output_coord: bsz, 4
        """
        output_cls = torch.mean(torch.stack([cls0, cls1, cls2]), dim=0)
        output_coord = self.bbox_head(output_cls).sigmoid()

        return output_coord

    def forward(
        self,
        image,
        lidar,
        text_ids,
        text_atts,
        verb_non_tag_labels,
        context_tag_labels,
        target_bbox=None,
    ):
        lidar = self.lidar_fc(lidar).reshape(-1, 32, 768)
        lidar1 = self.deformable1(lidar)
        lidar2 = self.deformable2(lidar)
        image_embeds, _ = self.get_vision_embeds(image)
        # image_embeds = torch.cat([image_embeds, lidar], dim=1)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        cross_embds = self.get_cross_embeds(
            image_embeds,
            torch.ones(image_embeds.shape[:2]).to(image_embeds.device),
            text_embeds=text_embeds,
            text_atts=text_atts,
        )
        cross_embds1 = torch.clone(cross_embds)
        cross_embds2 = torch.clone(cross_embds)
        cross_embds1[:, 0, :] = cross_embds1[:, 0, :] + lidar1[:, 0, :]

        tag_loss1, output_coord1, output_cls1 = self.recursive_layer1(
            cross_embds1, verb_non_tag_labels
        )
        cross_embds2[:, 0, :] = lidar2[:, 0, :]
        tag_loss2, output_coord2, output_cls2 = self.recursive_layer2(
            cross_embds2, context_tag_labels
        )
        # predict_emb  = (cross_embds[:,0,:] + cross_embds1[:,0,:] + cross_embds2[:,0,:]) / 3
        # output_coord = self.predict_final_bbox(
        #     cross_embds[:, 0, :], output_cls1, output_cls2
        # )
        output_coord = self.predict_final_bbox(
            cross_embds[:, 0, :], cross_embds1[:, 0, :], cross_embds2[:, 0, :]
        )
        if target_bbox is None:
            return output_coord
        # output_coord & target_bbox: 64, 4
        loss_bbox1, loss_giou1 = self.get_bbox_loss(output_coord1, target_bbox)
        loss_bbox2, loss_giou2 = self.get_bbox_loss(output_coord2, target_bbox)

        loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox)

        return (
            output_coord,
            loss_bbox,
            loss_giou,
            loss_bbox1,
            loss_giou1,
            loss_bbox2,
            loss_giou2,
            tag_loss1,
            tag_loss2,
        )
