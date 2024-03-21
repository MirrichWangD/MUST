# -*- coding: UTF-8 -*-

# 导入基本库
import io
import os
import base64
import json
import yaml
import time
import random
import warnings
import argparse
import itertools
from copy import deepcopy
from pathlib import Path

# 导入运算库
import torch
import torchvision.transforms as T
import numpy as np
from panopticapi.utils import id2rgb, rgb2id

# 导入可视化相关库
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# 导入自定义模块
import util.misc as utils
from models import build_model

try:
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    USE_D2 = True
except:
    USE_D2 = False

warnings.filterwarnings("ignore")


def get_args_parser():
    def str2bool(str):
        return True if str.lower() == 'true' else False

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config', default="configs/config.yaml", type=str,
                        help="配置文件路径")


    # 推理、后处理结果是否保存
    parser.add_argument("--save_result", default=True, type=str2bool,
                        help="是否保存模型推理的结果")
    # 图片设置
    parser.add_argument("--infer_img", default="assets/18002.jpg", type=str,
                        help="推理图片路径或目录路径")
    parser.add_argument("--img_ext", default="jpg,png,jpeg",
                        help="推理图片的类型")
    parser.add_argument("--output", default="output/inference", type=str,
                        help="推理结果保存目录路径")

    return parser


def main(args):
    assert args.config
    print(args)

    config = deepcopy(args)
    # 读取 config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_data = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in cfg_data.items():
            config.__setattr__(key, value)
    # 设置使用硬件
    device = torch.device(config.device)

    # 设置随机数种子
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 初始化模型，并且转移到硬件
    model, criterion, postprocessors = build_model(config)
    model.to(device)

    # 载入权重
    if config.resume:
        if config.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    infer_img = Path(args.infer_img)
    output_dir = Path(args.output)

    # 若推理参数是目录，则遍历允许后缀的图片文件名
    if infer_img.is_dir():
        output_dir = output_dir / infer_img.name
        infer_imgs = []
        for ext in args.img_ext.split(","):
            infer_imgs.extend(infer_img.glob("*." + ext))
        # print(infer_imgs)
    else:
        output_dir = output_dir / os.path.splitext(infer_img.name)[0]
        infer_imgs = [infer_img]

    # 自动创建输出文件夹
    output_dir.mkdir(parents=True, exist_ok=True)

    # 图像预处理
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 调用 Seaborn 的调色板
    palette = itertools.cycle(sns.color_palette())

    infer_time = []  # 图片推理时间
    results = dict()
    with tqdm(total=len(infer_imgs)) as pbar:
        for infer_img in infer_imgs:
            # 打开原始图片
            im = Image.open(infer_img).convert("RGB")
            # print(np.array(im).shape)
            img = transform(im).unsqueeze(0).to(device)  # 转换成张量

            # 模型推理
            t1 = time.time()
            out = model(img)
            t2 = time.time()
            infer_time.append(t2 - t1)
            # 后处理
            result = postprocessors["panoptic"](out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]
            # print(out)
            # print("=" * 50)
            # print(result)
            # 进行可视化
            panoptic_seg = Image.open(io.BytesIO(result['png_string']))
            panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
            # 定义RGB三通道矩阵 -> 全景分割ID矩阵
            panoptic_seg_id = rgb2id(panoptic_seg)

            # 将后处理结果的png_string转换成base64编码字符串
            result["png_string"] = base64.b64encode(result["png_string"]).decode('utf8')
            results[infer_img.name] = result

            if USE_D2:
                segments_info = deepcopy(result["segments_info"])
                panoptic_seg = torch.from_numpy(panoptic_seg_id)
                final_h, final_w = panoptic_seg.shape
                meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
                for i in range(len(segments_info)):
                    c = segments_info[i]["category_id"]
                    segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i][
                        "isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

                v = Visualizer(np.array(im.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
                v._default_font_size = 12
                v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
                img_output = Image.fromarray(v.get_image())
            else:
                # 对每一个mask进行上色
                panoptic_seg[:, :, :] = 0
                for id in range(panoptic_seg_id.max() + 1):
                    panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255

                img_output = Image.fromarray(panoptic_seg)
                # 混合原始图和分割图
                img_output = Image.blend(im, img_output, .5)

            if output_dir:
                img_output.save(output_dir / "result.png")
            else:
                plt.imshow(img_output)
                plt.axis('off')
                plt.title(infer_img.name)
                plt.tight_layout()
                plt.show()

            pbar.set_postfix_str(infer_img.name)
            pbar.update(1)

    # 保存推理结果到json文件
    if args.save_result:
        with open(output_dir / "result.json", "w+", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    print("Total time：%.4fs" % np.sum(infer_time))
    print("Per time：%.4fs" % np.mean(infer_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Panoptic Transformer Segmentation Inference Script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
