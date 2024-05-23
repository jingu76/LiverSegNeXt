# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from monai.data import SmartCacheDataset

import numpy as np
import torch

import monai

from monai import data, transforms
from monai.data import load_decathlon_datalist


class Sampler(torch.utils.data.Sampler):
    # 数据的分布式采样
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        #num_replicas:分布式的进程数
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        # 获取分布式训练中当前进程的排名
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        #将数据采样数使其均匀分布，以便每个进程得到相等数量的数据。
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # 每个进程的样本数量
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        # 获取每个进程对应的样本有效长度,total_size可能大于len(self.dataset),但是数组切片只会在不发生越界的前提下返回尽可能多的元素
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])
    # 定义一个迭代器
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            # 返回一个列表,[0,1,2...dataset长度-1]
            indices = list(range(len(self.dataset)))
        # 将数据补齐，让每个进程分配的样本数都相同
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    #选取前几个元素填充
                    indices += indices[: (self.total_size - len(indices))]
                # 如果len(indices)=1,则total_size=4,就不满足
                else:
                    #在[low,high)之间生成size个整数
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        # 应该是迭代dataset[indices]的元素
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_loader_mp(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image", "label"]), # 根据数据类型选择对应的读取器读取数据
            transforms.LoadImaged(keys=["image"],dtype=np.float32),
            transforms.LoadImaged(keys=["label"],dtype=np.uint8),
            transforms.EnsureChannelFirstd(keys=["image", "label"]), # 把series维度变成channel
            # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"), # 统一图像方向
            # transforms.Spacingd( # 按照pixdim对图像进行重采样
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            # transforms.ScaleIntensityRanged( # 图像值变化a->b（类似clip）
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"), # 矩形裁剪，按照值>0
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.RandCropByPosNegLabeld(  # 按照特定阴性阳性比例裁剪子图
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),  # 随机水平翻转
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),  # 随机旋转
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),  # 随机放大图像值
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),  # 随机偏移图像值
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image", "label"]),
            transforms.LoadImaged(keys=["image"],dtype=np.float32),
            transforms.LoadImaged(keys=["label"],dtype=np.uint8),
            transforms.EnsureChannelFirstd(keys=["image", "label"]), # 把series维度变成channel
            # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            # transforms.Spacingd(
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            # transforms.Spacingd(
            #     keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear")
            # ),
            # transforms.ScaleIntensityRanged(
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image", "label"]),
            transforms.LoadImaged(keys=["image"],dtype=np.float32),
            transforms.LoadImaged(keys=["label"],dtype=np.uint8),
            transforms.EnsureChannelFirstd(keys=["image", "label"]), # 把series维度变成channel
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            # transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            # transforms.ScaleIntensityRanged(
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            transforms.SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        # test_files = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            # persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.debug:
            datalist=datalist[:5]
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
            # train_ds = SmartCacheDataset(data=datalist, transform=train_transform, cache_rate=1.0,num_init_workers=args.workers,num_replace_workers=args.workers)

        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            # pin_memory=True,
            pin_memory=False,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            # val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
        )
        loader = [train_loader, val_loader]

    return loader
