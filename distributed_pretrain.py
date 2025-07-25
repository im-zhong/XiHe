# 2025/7/24
# zhangzhong


# 我们希望怎么使用这个脚本呢？
# pretrain.py --config config.yaml
# 剩下的所有配置都写在配置文件里面就行了呗
# 必要的时候，可以提供一些额外的参数，用来覆盖config里面的配置足够了


from torch.optim.lr_scheduler import LambdaLR
from xihe.model import Transformer
from xihe.settings import load_config
from xihe.trainer import (
    DistributedGPTTrainer,
    create_optimizer,
    create_cosine_lr_scheduler,
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

# torch.multiprocessing is a PyTorch wrapper around Python’s native multiprocessing
import torch.multiprocessing as mp
from typing import Any
import random
import numpy as np


def create_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)


def set_all_rng_states(rng_states):
    random.setstate(rng_states["python"])
    np.random.set_state(rng_states["numpy"])
    torch.set_rng_state(rng_states["torch_cpu"])
    if rng_states["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(rng_states["torch_cuda"])


def train_from_scratch(rank, world_size, config: Config):
    # TODO: tmp set this code, if we impl ddp, should delete this
    # ? 这设置的不是torch 创建tensor的默认设备？
    # 这个并不是用来设置默认设备的，pytorch没有提供这个功能
    # 这个是设置默认的cuda设备的，也就是当你指定 cuda 但是没有指定 cuda:id 的时候的默认id
    torch.cuda.set_device(rank)  # Set the default CUDA device to 0

    # Load configuration
    # config = load_config(Path(args.conf))
    # 在这里创建wandb更好
    wandb.login()
    run: Run = wandb.init(
        project="My LLM",
        config=config.model_dump(),  # Track hyperparameters and metadata
    )

    # get tokenizer from tokenizer configs
    # 但是我们不能直接依赖TokenizerConfig这个类
    # 要写一个函数来根据配置获取tokenizer
    # tokenizer = get_tokenizer(config.tokenizer)
    tokenizer = create_tokenizer(config.tokenizer.tokenizer_name)

    # 还需要设置dataset
    # 我们使用哪些dataset，和使用多少，在datasetconfig里面写上就行了
    datasets = [
        create_dataset(
            path=dataset.path,
            name=dataset.name,
            split=dataset.split,
        )
        for dataset in config.dataloader.datasets
    ]

    sampling_probabilities: list[float] = []
    if config.dataloader.sampling_probabilities:
        sampling_probabilities = config.dataloader.sampling_probabilities
    else:
        # 如果没有提供采样概率，就计算一下
        sampling_probabilities = calculate_sampling_probabilities(
            pathes=[dataset.path for dataset in config.dataloader.datasets],
            names=[dataset.name for dataset in config.dataloader.datasets],
            num_epochs=[dataset.num_epochs for dataset in config.dataloader.datasets],
        )

    # # Initialize dataset and dataloader
    dataset = PackingDataset(
        datasets=datasets,
        tokenizer=tokenizer,
        sampling_probabilities=sampling_probabilities,
    ).to_distributed_dataset(
        batch_size=config.dataloader.batch_size,
        context_length=config.model.context_length,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.dataloader.batch_size,
        # sampler=sampler,
    )

    # # Initialize model
    # TODO: vocab size应该是不用设置的
    # 应该可以通过tokenizer对象来获取才对
    model = Transformer(
        vocab_size=config.tokenizer.vocab_size,
        max_seq_len=config.model.context_length,
        num_layers=config.model.num_layers,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        intermediate_size=config.model.intermediate_size,
        device=config.trainer.device,  # Pass the device from config
    )
    # model = model.to(config.trainer.device)

    # 这里需要创建optimizer和scheduler
    # TODO: 这些东西都得重写，optimizer必须在模型的参数放在正确的设备上才能创建
    optimizer: Optimizer = create_optimizer(
        name=config.optimizer.optimizer_name,
        learning_rate=config.optimizer.initial_lr,
        weight_decay=config.optimizer.weight_decay,
        parameters=model.parameters(),
    )
    lr_scheduler: LambdaLR = create_cosine_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=config.trainer.warmup_steps,
        total_steps=config.trainer.total_steps,
        initial_lr=config.optimizer.initial_lr,
        max_lr=config.optimizer.max_lr,
        final_lr=config.optimizer.final_lr,
    )

    # # Initialize the trainer
    trainer = DistributedGPTTrainer(
        vocab_size=config.tokenizer.vocab_size,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        dataloader=dataloader,
        device=config.trainer.device,
        # dtype=config.model.dtype,
        run=run,
    )

    # Start training
    trainer.train(rank, world_size)


# 感觉这里传入config对象更好
# def train_from_scratch(rank: int, world_size: int, conf: Config):
#     pass


# 这个传入一个checkpoint对象更好吧
# 咱们先不直接写checkpoint了，先用最简单的方式把这个函数写出来
# 再看看能不能重构一个checkpoint出来
def train_from_checkpoint(rank: int, world_size: int, checkpoint: dict[str, Any]):
    torch.cuda.set_device(rank)  # Set the default CUDA device to 0
    wandb.login()

    set_all_rng_states(checkpoint["rng_state"])

    # 需要设计一种文件结构
    # 不对啊，其实没有任何的必要
    # 我们就全部放在一个大的dict里面就行了
    # 这样最灵活了，所以这应该是一个file而不是一个dir
    # checkpoint: dict[str, Any] = torch.load(ckpt_file)
    # resume config first
    # TODO: 这里的每一步都应该可以被测试
    config: Config = Config(**checkpoint["config"])

    tokenizer = create_tokenizer(config.tokenizer.tokenizer_name)

    # then load wandb
    run: Run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        id=config.wandb.id,
        resume="must",
    )

    # load model
    # define model first
    model = Transformer(
        vocab_size=config.tokenizer.vocab_size,
        max_seq_len=config.model.context_length,
        num_layers=config.model.num_layers,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        intermediate_size=config.model.intermediate_size,
        device=config.trainer.device,  # Pass the device from config
    )
    # model = checkpoint.load_model(model)
    model.load_state_dict(checkpoint["model"])
    model = model.to(rank)

    # TODO: 这个还需要研究
    # load dataloader
    # 那就只需要构建dataset
    # 生成dataloader
    # 然后load state就行了

    datasets = [
        create_dataset(
            path=dataset.path,
            name=dataset.name,
            split=dataset.split,
        )
        for dataset in config.dataloader.datasets
    ]

    sampling_probabilities: list[float] = []
    if config.dataloader.sampling_probabilities:
        sampling_probabilities = config.dataloader.sampling_probabilities
    else:
        # 如果没有提供采样概率，就计算一下
        sampling_probabilities = calculate_sampling_probabilities(
            pathes=[dataset.path for dataset in config.dataloader.datasets],
            names=[dataset.name for dataset in config.dataloader.datasets],
            num_epochs=[dataset.num_epochs for dataset in config.dataloader.datasets],
        )

    # # Initialize dataset and dataloader
    dataset = PackingDataset(
        datasets=datasets,
        tokenizer=tokenizer,
        sampling_probabilities=sampling_probabilities,
    )
    dataloader = dataset.to_stateful_dataloader(
        batch_size=config.dataloader.batch_size,
        context_length=config.model.context_length,
        rank=rank,
        world_size=world_size,
    )
    # TODO: 这些特定的save和load逻辑都应该封装在一起
    # 方便阅读理解和测试
    dataloader.load_state_dict(checkpoint["dataloader"][rank])

    grad_scaler = torch.amp.GradScaler("cuda", enabled=config.trainer.use_amp)
    grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # load optimizer
    optimizer: Optimizer = create_optimizer(
        name=config.optimizer.optimizer_name,
        learning_rate=config.optimizer.initial_lr,
        weight_decay=config.optimizer.weight_decay,
        parameters=model.parameters(),
    )
    optimizer.load_state_dict(checkpoint["optimizer"])

    # load scheduler
    lr_scheduler: LambdaLR = create_cosine_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=config.trainer.warmup_steps,
        total_steps=config.trainer.total_steps,
        initial_lr=config.optimizer.initial_lr,
        max_lr=config.optimizer.max_lr,
        final_lr=config.optimizer.final_lr,
    )
    lr_scheduler.load_state_dict(checkpoint["scheduler"])

    # # Initialize the trainer
    # TODO: 这里感觉也不是很好
    # 模型是不是应该只加载一次？然后由DDP负责将模型复制到每个GPU上？
    # 现在的写法，每个rank都会创建一个新的模型实例，都会加载之前的权重
    # 然后DDP又复制了一次
    # 不过这个点感觉消耗的时间并不多，所以先这样吧
    trainer = DistributedGPTTrainer(
        config=config,
        vocab_size=config.tokenizer.vocab_size,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        dataloader=dataloader,
        device=config.trainer.device,
        # dtype=config.model.dtype,
        grad_scaler=grad_scaler,
        rank=rank,
        world_size=world_size,
        run=run,
    )

    # Start training
    trainer.train(rank, world_size)


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
        checkpoint: dict[str, Any] = torch.load(ckpt_dir)
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
