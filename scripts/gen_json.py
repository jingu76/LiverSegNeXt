'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-10-13 06:38:14
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-28 06:56:53
FilePath: /SwinUNETR2Med/scripts/gen_json.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import os
import json
from tqdm import tqdm

basic_info = {
    "description": "liver tumor",
    "labels": {
        "0": "background",
        "1": "Cancer",
        "2": "Hemangioma",
        "3": "Cyst",
    },
    "licence": "yt",
    "modality": {
        "0": "CT"
    },
    "name": "liver_tumor",
    "numTest": 0,
    "numTraining": 0,
    "reference": "Zhejiang University",
    "release": "1.0 23/08/2023",
    "tensorImageSize": "3D",
    "test": [],
    "training": [], # need
    "validation": [], # need
}

def get_list(root_dir, data_dir):
    results = []
    
    data_dir_abs = os.path.join(root_dir, data_dir)
    images_dir_abs = os.path.join(data_dir_abs, "images")
    labels_dir_abs = os.path.join(data_dir_abs, "labels")
    
    image_names = os.listdir(images_dir_abs)
    for image_name in tqdm(image_names):
        # if "phase_2" not in image_name:
        #     continue
        label_name = image_name.replace('CT', 'SEG')
        label_path_abs = os.path.join(labels_dir_abs, label_name)
        if not os.path.exists(label_path_abs):
            raise Exception(f"{label_path_abs} not exist")
        
        image_path = os.path.join(data_dir, "images", image_name)
        label_path = os.path.join(data_dir, "labels", label_name)
        results.append({"image":image_path, "label":label_path})
    
    return results

def main(args):
    data = basic_info.copy()
    # 获得训练集对应的{image,label}路径
    data["training"] = get_list(args.root_dir, args.train_dir)
    data["validation"] = get_list(args.root_dir, args.val_dir)
    
    with open(args.output_path, "w") as f:
        json.dump(data, f)
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument("--root_dir", default="/data1/hhy/liver/liver/", type=str, help="root directory")
    parser.add_argument("--train_dir", default="train", type=str, help="input dataset directory")
    parser.add_argument("--val_dir", default="val", type=str, help="output directory")
    # parser.add_argument('--phase_idx', type=int, default=2)
    parser.add_argument("--output_path", default="/data1/hhy/liver/liver/dataset.json")
    args = parser.parse_args()
    main(args)