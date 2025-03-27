#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPose benchmark a recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--bz', default=32, type=int, help='test config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # Since we only care about the forward speed of the network
    cfg.model.pretrained=None
    cfg.model.test_cfg.flip_test=False
    cfg.model.test_cfg.use_udp=False
    cfg.model.test_cfg.post_process='none'

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.bz,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # get the example data
    for i, data in enumerate(data_loader):
        break

    # the first several iterations may be very slow so skip them
    num_warmup = 100
    inference_times = 100

    with torch.no_grad():
        start_time = time.perf_counter()
      
        for i in range(num_warmup):
            torch.cuda.synchronize()            
            model(return_loss=False, **data)
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
        print(f'warmup cost {elapsed} time')

        start_time = time.perf_counter()
        
        for i in range(inference_times):
            torch.cuda.synchronize()
            model(return_loss=False, **data)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        fps = args.bz * inference_times / elapsed
        print(f'the fps is {fps}')


if __name__ == '__main__':
    main()
