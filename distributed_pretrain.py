# 2025/7/24
# zhangzhong

# 我们希望怎么使用这个脚本呢？
# pretrain.py --config config.yaml
# 剩下的所有配置都写在配置文件里面就行了呗
# 必要的时候，可以提供一些额外的参数，用来覆盖config里面的配置足够了

import os
from torch.optim.lr_scheduler import LambdaLR
from xihe.model import Transformer
from xihe.settings import load_config
from xihe.trainer import (
    DistributedGPTTrainer,
)
from xihe.dataset import (
    PackingDataset,
    create_dataset,
    calculate_sampling_probabilities,
)
from pathlib import Path
import wandb
from wandb.sdk.wandb_run import Run
import argparse
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.optim import Optimizer, Adam, AdamW
from torch.utils.data import DataLoader
from xihe.settings import Config
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp  # torch.multiprocessing is a PyTorch wrapper around Python’s native multiprocessing
from typing import Any
import random
import numpy as np
from xihe.ckpt import Checkpoint, load_ckpt_from_path


# 这样整体上好一些
def train_from_scratch(rank: int, world_size: int, config: Config) -> None:
    trainer = DistributedGPTTrainer(
        rank=rank,
        world_size=world_size,
        config=config,
    )
    trainer.train(config)


def train_from_checkpoint(
    rank: int,
    world_size: int,
    checkpoint: Checkpoint,
) -> None:
    trainer = DistributedGPTTrainer(
        rank=rank,
        world_size=world_size,
        config=checkpoint.get_config(),
    )
    trainer.train(config, ckpt=checkpoint)


# 这样这里的代码就非常简单，就是配置 + 调用函数
# 这样最好
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a Transformer model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--conf", "-f", type=str, required=False, help="Path to the config file"
    )
    group.add_argument(
        "--ckpt", "-c", type=str, required=False, help="Path to the checkpoint dir"
    )
    args = parser.parse_args()
    conf_file = Path(args.conf) if args.conf else None
    ckpt_dir = Path(args.ckpt) if args.ckpt else None

    world_size = torch.cuda.device_count()
    print(f"Number of GPUs available: {world_size}", flush=True)

    # 对应着两种模式，一种是从头开始train，一种是从途中开始train
    # 要不写两个main吧，或者写两个train
    # 一个是train_from_scratch，一个是train_from_checkpoint
    if conf_file is not None:
        config = load_config(conf_file)
        mp.spawn(
            train_from_scratch,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
        )
        # train_from_scratch(conf_file)
    else:
        checkpoint: Checkpoint = load_ckpt_from_path(ckpt_dir)
        mp.spawn(
            train_from_checkpoint,
            args=(world_size, checkpoint),
            nprocs=world_size,
            join=True,
        )
    # TODO：移到train函数里面
    # world_size = torch.cuda.device_count()
    # print(f"Number of GPUs available: {world_size}", flush=True)
    # mp.spawn(
    #     main,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True,
    # )
