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

import argparse
import os

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.data_utils import get_loader,get_loader_mp
from utils.utils import cal_dice, resample_3d, unSpatialPad, info_if_main, reduce_by_weight

from monai.inferers import sliding_window_inference
from networks.MedNeXt import get_mednet

from loguru import logger

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")


parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0,1,2,3,4", type=str, help="")
parser.add_argument("--ngpus_per_node", default=1, type=int, help="gpu number")
parser.add_argument("--output_directory", type=str, help="output directory")

parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--logdir", default="/data4/ycy/experiment/test", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_pth",
    default="./log/model.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--kernel_size", default=3, type=int, help="kernel_size")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:12345", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=0, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--num_class", default=13, type=int, help='exclude background; if num_class=-1, calcuate binary dice')
parser.add_argument("--save_output", action="store_true", help='whether to save output as nifty file')
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")


def main():
    args = parser.parse_args()
    logger.add(os.path.join(args.logdir, 'test_log.txt'))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    if args.distributed:
        info_if_main(f"Found total gpus {args.ngpus_per_node}")
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    logger.add(os.path.join(args.logdir, "test_log.txt"))
    
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    
    args.test_mode = True
    if args.save_output:
        output_directory = args.output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    if args.in_channels>1:
        val_loader = get_loader_mp(args)
    else:
        val_loader = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth=args.pretrained_pth
    model = get_mednet(in_channels=args.in_channels,out_channels=args.out_channels,kernel_size=args.kernel_size)
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    # dice比较低的image,ct,dice
    # low_dice=[]
    dice_sum, data_num = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            input_shape = val_inputs.shape[2:]
            if input_shape[0] < args.roi_x or input_shape[1] < args.roi_y or input_shape[2] < args.roi_z:
                raise Exception("image size small than trainning roi")
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_outputs = unSpatialPad(val_outputs, val_inputs.cpu().numpy()[0, 0])
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            dice_list_sub = []
            if args.num_class == -1:
                organ_Dice = cal_dice(val_outputs > 0, val_labels > 0)
                dice_list_sub.append(organ_Dice)
                if  organ_Dice<0.95:
                    print(f"low dice:{organ_Dice},image_name:{img_name}")
            else:
                for i in range(1, args.num_class + 1):
                    organ_Dice = cal_dice(val_outputs == i, val_labels == i)
                    dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            logger.info("{}/{} {}: {}".format(i, len(val_loader), img_name, mean_dice))
            dice_sum += mean_dice
            data_num += 1
            # 修改下文件名
            img_name=img_name.replace('CT', 'PREDICT')
            if args.save_output:
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
                )
    avg_dice = reduce_by_weight(dice_sum/data_num, data_num)
    info_if_main("Overall Mean Dice: {}".format(np.mean(avg_dice)))

if __name__ == "__main__":
    main()
