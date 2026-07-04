import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from refTools.refer_python3 import REFER
from dataset.grounding_dataset import grounding_dataset_talk2car_lidar_bbox
from dataset.randaugment import RandomAugment


def create_dataset(dataset, config, evaluate=False):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if dataset == "grounding_bbox_talk2car_lidar":
        refer = REFER(
            data_root=config["refcoco_data"],
            dataset="talk2car",
            splitBy="google",
        )
        test_dataset = grounding_dataset_talk2car_lidar_bbox(
            test_transform,
            config["image_root"],
            mode="test",
            config=config,
            refer=refer,
        )
        if evaluate:
            return None, test_dataset

        train_transform = transforms.Compose(
            [
                RandomAugment(
                    2,
                    7,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Equalize",
                        "Brightness",
                        "Sharpness",
                    ],
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset = grounding_dataset_talk2car_lidar_bbox(
            train_transform,
            config["image_root"],
            mode="train",
            config=config,
            refer=refer,
        )
        return train_dataset, test_dataset

    raise ValueError(f"unknown dataset for VIGOR: {dataset}")


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(
            f"### be careful: func create_loader returns a list length of {len(loaders)}"
        )

    return loaders
