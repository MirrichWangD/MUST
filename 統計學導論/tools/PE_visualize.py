# -*- coding: UTF-8 -*-


# 导入模块
import math
import torch
import numpy as np

# 导入可视化模块
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = ["Microsoft YaHei"]

"""
=====================
@@@ Settings
=====================
"""

root = r"..\data\COCO\val2017"
out_dir = r"..\assets"
img1_id = "000000011197"  # 640x427
img2_id = "000000439623"  # 480x640

"""
====================
@@@ Masks 可视化
====================
"""

# 使用 PIL 读取图片
img1 = Image.open(rf"{root}\{img1_id}.jpg")
img2 = Image.open(rf"{root}\{img2_id}.jpg")
# 转换成矩阵
img1_arr = np.array(img1)
img2_arr = np.array(img2)
# 获取最大 H, W
H1, W1 = img1_arr.shape[:2]
H2, W2 = img2_arr.shape[:2]
H, W = max(H1, H2), max(W1, W2)
print("Mask H: %s, W: %s" % (H, W))

# 原始 Mask
mask_raw = np.ones((2, H, W), dtype=np.bool)
mask_raw[0, :H1, :W1] = 0
mask_raw[1, :H2, :W2] = 0
# 保存可视化结果
plt.matshow(mask_raw[0])
plt.axis(False)
plt.savefig(rf"{out_dir}\mask1.png", bbox_inches="tight", pad_inches=0)
plt.matshow(mask_raw[1])
plt.axis(False)
plt.savefig(rf"{out_dir}\mask2.png", bbox_inches="tight", pad_inches=0)

# Backbone 后 Mask
h1 = 14
w1 = h2 = 20
w2 = 15
h = H // 32
w = W // 32
mask_backbone = np.ones((2, h, w), dtype=np.bool)
mask_backbone[0, :h1, :w1] = 0
mask_backbone[1, :h2, :w2] = 0
# 保存可视化结果
plt.matshow(mask_backbone[0])
plt.axis(False)
plt.savefig(rf"{out_dir}\mask1_backbone.png", bbox_inches="tight", pad_inches=0)
plt.matshow(mask_backbone[1])
plt.axis(False)
plt.savefig(rf"{out_dir}\mask2_backbone.png", bbox_inches="tight", pad_inches=0)

"""
====================
@@@ 位置编码 PE 可视化
====================
"""

# 相关设置
is_normalize = False
hidden_dim = 256  # 输入向量维度
num_pos_feats = hidden_dim // 2  # 位置编码维度
temperature = 10000  # 特殊值
scale = 2 * math.pi
eps = 1e-6

# 使用特征图 Mask
mask = torch.from_numpy(mask_backbone)
not_mask = ~mask

y_embed = not_mask.cumsum(1, dtype=torch.float32)
x_embed = not_mask.cumsum(2, dtype=torch.float32)
if is_normalize:
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

pos_x = x_embed[:, :, :, None] / dim_t
pos_y = y_embed[:, :, :, None] / dim_t
pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

# 两种方向累积和可视化
fig, ax = plt.subplots(1, 2, figsize=(7, 3))
ax[0].imshow(x_embed[0].numpy())
# ax[0].axis(False)
ax[0].set_title("横坐标累积和")
ax[1].imshow(y_embed[0].numpy())
# ax[1].axis(False)
ax[1].set_title("横坐标累积和")
plt.tight_layout()
plt.show()
# -------------- 显示颜色版 ------------- #
# plt.matshow(x_embed[0].numpy())
# plt.colorbar()
# plt.show()
# -------------------------------------- #

# 位置编码可视化
plt.matshow(pos[0,0])
plt.show()