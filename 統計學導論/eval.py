# -*- coding: UTF-8 -*-

# 导入基本模块
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
import numpy as np
import argparse
import random
import torch
import yaml
# 导入自定义模块
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
from engine import evaluate


def get_args_parser():
    """ 设置对象解释器 """
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config', default="configs/config.yaml", type=str)
    return parser


def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)  # 获取训练硬件，cuda:0 | cpu
    # 生成模型、损失函数和后处理对象
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    # 判断是否使用分布式模式
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # 构建验证数据集
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 定义训练、验证数据迭代器
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # 判断是否为全景分割，否则使用检测格式
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # 格式化输出目录，创建输出目录文件夹
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 判断是否需要加载权重
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                          data_loader_val, base_ds, device, args.output_dir)
    if args.output_dir:
        utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR Segmentation Evaluate Script', parents=[get_args_parser()])
    args = parser.parse_args()
    # 读取configs/<config>.yaml文件，并且设置到args对象中
    with open(args.config, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            args.__setattr__(key, value)
    main(args)
