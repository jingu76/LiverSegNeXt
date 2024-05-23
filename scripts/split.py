import os
import random
import shutil

# 定义路径
source_folder = "/data4/train_liver_475"
train_folder = "/data1/hhy/new_liver/npz/train/"
val_folder = "/data1/hhy/new_liver/npz/val/"

# 确保目标文件夹存在
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# 获取所有npz文件的列表
npz_files = [f for f in os.listdir(source_folder) if f.endswith(".npz")]

# 随机打乱文件列表
random.shuffle(npz_files)

# 计算切分点，使得训练集:验证集=4:1
split_point = int(0.8 * len(npz_files))

# 划分训练集和验证集
train_files = npz_files[:split_point]
val_files = npz_files[split_point:]

# 将文件复制到目标文件夹
for file in train_files:
    source_path = os.path.join(source_folder, file)
    target_path = os.path.join(train_folder, file)
    shutil.copy(source_path, target_path)

for file in val_files:
    source_path = os.path.join(source_folder, file)
    target_path = os.path.join(val_folder, file)
    shutil.copy(source_path, target_path)

print("划分完成！")
