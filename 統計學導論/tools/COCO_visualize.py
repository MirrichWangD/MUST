# -*- coding: UTF-8 -*-

# 导入基本库
import os
import cv2
import json
import pandas as pd
import numpy as np
import pycocotools.mask as mask_utils
from collections import Counter

# 导入可视化库
from PIL import Image, ImageFont, ImageDraw
import imgviz
import matplotlib.pyplot as plt
import seaborn as sns

"""
======================
@@@ 自定义工具函数
======================
"""


def load_json(file):
    """读取 json 格式文件函数

    Args:
        file: [str] 文件目录字符串

    Returns:
        dict

    """
    with open(file, 'r') as f:
        data = json.load(f)
    return data


"""
======================
@@@ Config
======================
"""

np.random.seed(666)  # 随机数种子，使得实验可以复现
plt.rcParams["font.family"] = ["Microsoft YaHei"]  # mpl 使用微软雅黑字体draw = ImageDraw.Draw(img)
font = ImageFont.truetype("msyh.ttc", 10, encoding="utf-8")
colormap = imgviz.label_colormap()  # 获取图像颜色映射
root = r"..\data\COCO"  # 数据集路径
image_id = 11197  # 图片 id

"""
======================
@@@ 加载图片和标注文件
======================
"""

# 以2017年验证集为例子
instance = load_json(os.path.join(root, "annotations/instances_val2017.json"))  # 读取目标检测/实例分割 json
panoptic = load_json(os.path.join(root, "annotations/panoptic_val2017.json"))  # 读取全景分割 json
stuff = load_json(os.path.join(root, "annotations/stuff_val2017.json"))  # 读取语义分割 json

# 构造图片ID对应的标注信息
### 图片
images = {}
for line in instance["images"]:
    images[line["id"]] = line

image_ids = set(list(map(lambda i: i["image_id"], stuff["annotations"])))  # 图像ID
### 语义分割标注
stuff_anns = dict(zip(image_ids, [list() for _ in range(len(image_ids))]))
for line in stuff["annotations"]:
    stuff_anns[line["image_id"]].append(line)

### 实例分割标注
instance_anns = dict(zip(image_ids, [list() for _ in range(len(image_ids))]))
for line in instance["annotations"]:
    instance_anns[line["image_id"]].append(line)

### 全景分割标注
panoptic_anns = {}
for line in panoptic["annotations"]:
    panoptic_anns[line["image_id"]] = line

# 构造语义标签ID对应标签信息
stuff_labels = {}
for line in stuff["categories"]:
    stuff_labels[line["id"]] = line

# 构造实例标签ID对应标签信息
instance_labels = {}
for line in instance["categories"]:
    instance_labels[line["id"]] = line

"""
======================
@@@ 标注信息保存为csv文件
======================
"""

# pd.DataFrame(instance["images"]).to_csv(os.path.join(root, "COCO_images_val2017.csv"), index=False)
# for k in ["annotations", "categories"]:
#     pd.DataFrame(panoptic[k]).to_csv(os.path.join(root, f"COCO_panoptic_val2017_{k}.csv"), index=False)
#     pd.DataFrame(instance[k]).to_csv(os.path.join(root, f"COCO_instance_val2017_{k}.csv"), index=False)
#     pd.DataFrame(stuff[k]).to_csv(os.path.join(root, f"COCO_stuff_val2017_{k}.csv"), index=False)

# 获取图片信息
img_info = images[image_id]
img_name = os.path.splitext(img_info["file_name"])[0]  # 获取图片文件名
# 加载图片
img = Image.open(os.path.join(root, "val2017", f"{img_name}.jpg"))
img_stuff = Image.open(os.path.join(root, "annotations", "stuff_val2017_pixelmaps", f"{img_name}.png"))
img_panoptic = Image.open(os.path.join(root, "panoptic_val2017", f"{img_name}.png"))

# colormap = np.array(img_stuff.getpalette()).reshape(-1, 3)

W, H = img.size  # 获取图像尺寸

"""
======================
@@@ Stuff 可视化处理
======================
"""

# 语义标签和标注
stuff_category = pd.DataFrame(stuff["categories"])
stuff_anno = pd.DataFrame(stuff["annotations"])
stuff_anno = stuff_anno[stuff_anno["image_id"] == image_id].to_dict(orient="records")

# ------------------------------------------------------------------------------------ #
# 从 json 中提取分割 mask
mask_stuff = np.zeros((H, W), dtype=np.int32)
img_stuff_label = np.zeros((H, W, 3), dtype=np.uint8)
for line in stuff_anno:
    mask_stuff += mask_utils.decode(line["segmentation"]) * line["category_id"]
    img_stuff_label[mask_stuff == line["category_id"]] = colormap[line["category_id"]]
# img_stuff_label = Image.fromarray(mask_stuff.astype(np.uint8), "P")
# img_stuff_label.putpalette(colormap.flatten())
# ------------------------------------------------------------------------------------ #

# img_stuff_mix = Image.blend(img, img_stuff.convert("RGB"), 0.7)

"""
======================
@@@ Instance 可视化处理
======================
"""

# 实例标签和标注
instance_category = dict(pd.DataFrame(instance["categories"])[["id", "name"]].values)
instance_anno = pd.DataFrame(instance["annotations"])
instance_anno = instance_anno[instance_anno["image_id"] == image_id].to_dict(orient="records")

mask_instance = np.zeros((H, W, 3), dtype=np.uint8)  # 初始化全为0的mask矩阵
for line in instance_anno:
    color = np.random.randint(1, 256, 3).tolist()
    seg_pts = np.array(list(zip(line["segmentation"][0][::2], line["segmentation"][0][1::2])), dtype=np.int32)
    x, y, w, h = line["bbox"]
    cv2.polylines(mask_instance, [seg_pts], True, (0, 0, 0), 2)  # 画边界线
    cv2.fillPoly(mask_instance, [seg_pts], color)  # 生成区域掩码
    # cv2.rectangle(mask_instance, [int(x), int(y)], [int(x + w), int(y + h)], color, 2)

img_instance_mix = Image.blend(img, Image.fromarray(mask_instance), 0.7)
img_instance_label = Image.fromarray(mask_instance)

# draw = ImageDraw.Draw(img_instance_mix)
# for line in instance_anno:
#     category_name = instance_category[line["category_id"]]
#     x, y, w, h = line["bbox"]
#     draw.text((x, y - 16), f"{category_name}-{line['id']}", (255, 255, 255), font)

"""
======================
@@@ Panoptic 可视化处理
======================
"""

# 将全景标签转换成矩阵
img_panoptic_arr = np.array(img_panoptic, dtype=np.int32)
panoptic_category = dict(map(
    lambda i: (i["id"], {"isthing": i["isthing"], "name": i["name"]}),
    panoptic["categories"]
))
panoptic_anno_df = pd.DataFrame(panoptic["annotations"])
panoptic_anno = panoptic_anno_df[panoptic_anno_df["image_id"] == image_id]
# 全景标签
panoptic_ids = img_panoptic_arr[:, :, 0] + 256 * img_panoptic_arr[:, :, 1] + 256 ** 2 * img_panoptic_arr[:, :, 2]
panoptic_ids_map = dict(zip(np.unique(panoptic_ids), range(len(panoptic_ids))))
mask_panoptic = np.zeros((H, W, 3), dtype=np.int32)
for line in panoptic_anno["segments_info"].values[0]:
    category_id = line["category_id"]
    x, y, w, h = line["bbox"]
    is_thing = panoptic_category[category_id]["isthing"]
    # mask_panoptic[panoptic_ids == line["id"]] = panoptic_ids_map[line["id"]]
    if is_thing:
        # cv2.rectangle(img_panoptic_arr, [int(x), int(y)], [int(x + w), int(y + h)], colormap[category_id].tolist(), 2)
        mask_panoptic[panoptic_ids == line["id"]] = np.random.randint(0, 255, 3)
    else:
        color = colormap[category_id]
        if (color == 0).all():
            color = np.random.randint(0, 255, 3)
        mask_panoptic[panoptic_ids == line["id"]] = color

"""
======================
@@@ 结果可视化
======================
"""

fig, ax = plt.subplots(2, 2, figsize=(9, 6))
fig.suptitle("MS COCO 2017", fontsize=20)
# 原始图像
ax[0, 0].imshow(img)
ax[0, 0].set_title("原始图像")
ax[0, 0].axis(False)
# 语义标签
ax[0, 1].imshow(img_stuff)
ax[0, 1].set_title("语义分割标签")
ax[0, 1].axis(False)
# 实例分割混合图
ax[1, 0].imshow(img_instance_label)
ax[1, 0].set_title("实例分割标签")
ax[1, 0].axis(False)
# 全景标签 ID
ax[1, 1].imshow(img_panoptic)
ax[1, 1].set_title("全景分割标签")
ax[1, 1].axis(False)

fig.tight_layout()
fig.show()

for i, img in enumerate([img, img_stuff, img_instance_label, img_panoptic]):
    plt.imshow(img)
    plt.axis(False)
    plt.savefig(f"../assets/p2.{i+1}.png", bbox_inches="tight")

"""
=========================
@@@ 统计训练、验证集的标签数量
=========================
"""

train_instance = load_json(os.path.join(root, "annotations/instances_train2017.json"))

max_height = max(list(map(lambda i: i["height"], train_instance["images"])))
max_width = max(list(map(lambda i: i["width"], train_instance["images"])))
print("Max Height: %s, Width: %s"% (max_height, max_height))

# 训练集
# panoptic_train = load_json(os.path.join(root, "annotations", "panoptic_train2017.json"))
# train_categories = []
# for anno in panoptic_train["annotations"]:
#     tmp = list(map(lambda i: i["category_id"], anno["segments_info"]))
#     train_categories.extend(tmp)
# train_counts = dict(sorted(Counter(train_categories).items(), key=lambda i: int(i[0])))
#
# # 验证集
# val_categories = []
# for anno in panoptic["annotations"]:
#     tmp = list(map(lambda i: i["category_id"], anno["segments_info"]))
#     val_categories.extend(tmp)
# val_counts = Counter(val_categories)
#
# instance_train = load_json(os.path.join(root, "annotations", "instances_train2017.json"))
# train_cate = list(map(lambda i: i["category_id"], instance_train["annotations"]))