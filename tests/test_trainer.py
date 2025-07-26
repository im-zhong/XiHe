# 2025/7/26
# zhangzhong

from xihe.trainer import BasicGPTTrainer
from pathlib import Path
from xihe.model import Transformer
from xihe.optimizer import create_optimizer, create_cosine_lr_scheduler
from xihe.dataset import create_dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Any
from xihe.tokenizer import create_tokenizer
from xihe.dataset import create_dataloader
from xihe.schemas import DatasetArgs
from torch.optim.lr_scheduler import LambdaLR
import torch
from xihe.settings import Config
from tests.common import generate_testing_config
from xihe.trainer import DistributedGPTTrainer


def test_basic_gpt_trainer():
    tokenizer = create_tokenizer("gpt2")
    print(f"Tokenizer: {tokenizer}")

    rank = 0
    world_size = 4
    batch_size = 4
    context_length = 1024
    # dataloader = create_dataloader(
    #     tokenizer=tokenizer,
    #     rank=rank,
    #     batch_size=batch_size,
    #     context_length=context_length,
    #     world_size=world_size,  # Assuming single GPU for this test
    #     datasets_args=[
    #         DatasetArgs(
    #             path="allenai/c4",
    #             name="en",
    #             split="train[:1024]",
    #             num_epochs=1,
    #             streaming=False,
    #         ),
    #         DatasetArgs(
    #             path="wikimedia/wikipedia",
    #             name="20231101.en",
    #             split="train[:1024]",
    #             num_epochs=2,
    #             streaming=False,
    #         ),
    #     ],
    # )
    # print(f"Dataloader: {dataloader}")

    num_layers = 2
    hidden_size = 512
    num_heads = 8
    intermediate_size = 2048
    device = "cpu"  # For testing, we can use CPU
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=context_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        device=device,  # Pass the device from config
    )
    model = model.to(device)
    print(f"Model: {model}")

    optimizer_name = "adamw"
    initial_lr = 1e-4
    weight_decay = 0.01
    optimizer: Optimizer = create_optimizer(
        name=optimizer_name,
        learning_rate=initial_lr,
        weight_decay=weight_decay,
        parameters=model.parameters(),
    )
    print(f"Optimizer: {optimizer}")

    warmup_steps = 1000
    total_steps = 10000
    max_lr = 1e-3
    final_lr = 1e-5

    lr_scheduler: LambdaLR = create_cosine_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        initial_lr=initial_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )
    print(f"LR Scheduler: {lr_scheduler}")

    grad_scaler = torch.amp.GradScaler()  # Enable mixed precision training

    trainer = BasicGPTTrainer(
        vocab_size=tokenizer.vocab_size,
        # dataloader=dataloader,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    # 不对啊，如果可以指定数据，为什么要引入dataloader呢？
    trainer.train_step(
        step=0,
        batch={
            "input_ids": torch.randint(
                0, tokenizer.vocab_size, (batch_size, context_length)
            )
        },
    )

    state_dict = trainer.get_state_dict(step=0)
    print(f"Trainer state dict: {state_dict}")


def test_distributed_trainer_from_scratch():
    # 那么这里就要先构造一个config对象
    # 咱们最好不要从example_conf来读
    # 就在这里构造一个config对象就行了 减少对外部的依赖
    config = generate_testing_config()
    print(f"Config: {config}")
    rank = 0
    world_size = 1
    trainer = DistributedGPTTrainer(
        rank=rank,
        world_size=world_size,  # For testing, we can use a single process
        run=None,  # No wandb run for this test
        config=config,  # Pass the config directly
    )

    trainer.train_from_scratch(rank=rank, world_size=world_size, config=config)


def test_distributed_trainer_from_ckpt():
    pass
