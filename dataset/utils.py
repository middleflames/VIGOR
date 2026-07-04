"""Dataset helpers for VIGOR talk2car grounding (trimmed from X-VLM dataset.utils).

Keeps only the caption/box helpers used by the grounding pipeline; the VQA,
refcoco RefEvaluation and COCO-caption evaluators were dropped.
"""

import os
import re

import torch
import torch.distributed as dist
from tqdm import tqdm

import utils


def pre_caption(caption, max_words):
    caption = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
        .replace("<person>", "person")
    )

    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    if not len(caption):
        raise ValueError("pre_caption yields invalid text")

    return caption


def collect_tensor_result(result, filename, local_wdir):
    wpath = os.path.join(local_wdir, "%s_rank%d.pth" % (filename, utils.get_rank()))
    torch.save(result, wpath)

    dist.barrier()

    result = []
    if utils.is_main_process():
        # combine results from all processes
        for rank in range(utils.get_world_size()):
            rpath = os.path.join(local_wdir, "%s_rank%d.pth" % (filename, rank))
            result += torch.load(rpath)

    dist.barrier()

    return result


def grounding_eval_talk2car_bbox(results, refer):
    correct_A_d_50, correct_val_d_50 = 0, 0
    correct_A_d_75, correct_val_d_75 = 0, 0
    correct_A_d_90, correct_val_d_90 = 0, 0
    correct_A_d_30, correct_val_d_30 = 0, 0
    correct_A_d_10, correct_val_d_10 = 0, 0
    num_test, num_val = 0, 0 
    total_iou_val = 0.0
    total_iou_test = 0.0
    IOU_list = []
    for res in tqdm(results):
        ref_id = res["ref_id"]
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]["bbox"]
        image = refer.Imgs[ref["image_id"]]

        coord = res["pred"].cuda()
        coord[0::2] *= image["width"]
        coord[1::2] *= image["height"]

        coord[0] -= coord[2] / 2
        coord[1] -= coord[3] / 2

        IoU_det = computeIoU(ref_box, coord)
        res['iou'] = IoU_det.item()
        IOU_list.append(IoU_det)
        if ref["split"] == "test":
            num_test += 1
            total_iou_test += IoU_det.item()
            if IoU_det >= 0.1:
                correct_A_d_10 += 1
            if IoU_det >=0.3:
                correct_A_d_30 += 1
            if IoU_det >= 0.5:
                correct_A_d_50 += 1
            if IoU_det >= 0.75:
                correct_A_d_75 += 1
            if IoU_det >= 0.9:
                correct_A_d_90 += 1
        elif ref["split"] == "val":
            num_val += 1
            total_iou_val += IoU_det.item()
            if IoU_det >= 0.1:
                correct_val_d_10 += 1
            if IoU_det >= 0.3:
                correct_val_d_30 += 1
            if IoU_det >= 0.5:
                correct_val_d_50 += 1
            if IoU_det >= 0.75:
                correct_val_d_75 += 1
            if IoU_det >= 0.9:
                correct_val_d_90 += 1
            if IoU_det >= 0.5:
                correct_val_d_50 += 1
    # eval_result = {"val_d": correct_val_d / num_val, "testA_d": correct_A_d / num_test}
    eval_result = {"val_d10": correct_val_d_10 / num_test, "testA_d10": correct_A_d_10 / num_test,
                   "val_d30": correct_val_d_30 / num_test, "testA_d30": correct_A_d_30 / num_test,
                   "val_d": correct_val_d_50 / num_test, "testA_d":correct_A_d_50 / num_test,
                   "val_d75": correct_val_d_75 / num_test,"testA_d75":correct_A_d_75 / num_test,
                   "val_d90": correct_val_d_90 / num_test, "testa_d90":correct_A_d_90 /num_test,
                   "test_iou": total_iou_test / num_test}

    for metric, acc in eval_result.items():
        print(f"{metric}: {acc:.3f}")
    return eval_result, IOU_list


def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return float(inter) / union
