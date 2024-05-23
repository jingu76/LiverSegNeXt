# 肝肿瘤数据集分割准确率度量Python脚本

## 主要功能

本脚本主要用于计算肝肿瘤分割结果准确率，通过度量分割图与真实标注之间的差异实现。

## 数据源要求

本脚本要求输入数据为指定数据集的分割结果文件和真实标注mask文件。并需要符合以下要求：
1. 使用内部多期npz格式
2. 分割结果和标注文件分别存放于两个不同的文件夹
3. 不同文件夹下，同一份CT的标注文件和分割结果文件名要相同

## 使用方法与传参规则

脚本通过传入必要的6个参数使用，可以直接启动，也可以通过命令行启动。

脚本需要传入的参数共有6个

1. '--pre_dir'：分割结果文件夹路径
2. '--gt_dir'：真实标注文件夹路径
3. '--num_classes'：肿瘤分类数量，是三分类还是五分类。只能传3或5，默认为3
4. '--series_num'：要测算哪个期，数值范围0-3
5. '--parallel'： 是否开启多线程模式，默认True开启
6. '--thread_num'：线程数量，默认为20

如果不采用命令行直接启动本脚本，可以在如下代码段中设定参数值：
```
parser.add_argument('--pre_dir', type=str, default=r'/data0/wulei/train_log/segmentor_dual_p_d/pred-28-post')
parser.add_argument('--gt_dir', type=str, default=r"/data0/datasets/liver_CT4_Z2_ts2.0")
parser.add_argument('--num_classes', type=int, default=3, choices=(3, 5))
parser.add_argument('--series_num', type=int, default=3, choices=(0, 1, 2, 3))
parser.add_argument('--parallel', type=bool, default=True, help='多进程模式，默认开启')
parser.add_argument('--thread_num', type=int, default=20, help='只有启动多进程模式（parallel为True）才有效')
```
修改对应值的default即可。

若采用命令行启动，可以使用如下方式：

```
python hi_measure.py --pre_dir '\xx\xx' --gt_dir '\yy\yy' ...
```

## 输出意义
肝分割总Dice | 肿瘤分割总Dice | 各类肿瘤Dice | 肿瘤总Recall | 肿瘤总Precision | 各类肿瘤Recall | 各类肿瘤Precision

多类肿瘤指标顺序按照文件定义的肿瘤类型顺序

