# 2025/7/20
# zhangzhong


# use pytorch lambdaLR to impl custom learning rate scheduler
import math
from collections.abc import Callable

from torch.optim import (
    Adam,
    AdamW,  # use this optimizer
    Optimizer,
)

# https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import ParamsT


def create_optimizer(
    name: str, learning_rate: float, weight_decay: float, parameters: ParamsT
) -> Optimizer:
    if name.lower() == "adamw":
        return AdamW(params=parameters, lr=learning_rate, weight_decay=weight_decay)
    if name.lower() == "adam":
        return Adam(params=parameters, lr=learning_rate, weight_decay=weight_decay)
    msg = f"Unsupported optimizer: {name}"
    raise ValueError(msg)


def cosine_scheduler_with_warmup(
    warmup_steps: int,
    total_steps: int,
    initial_lr: float,
    max_lr: float,
    final_lr: float,
) -> Callable[..., float]:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return initial_lr + (max_lr - initial_lr) * step / warmup_steps
        # step >= lr
        theta: float = math.pi * (step - warmup_steps) / (total_steps - warmup_steps)
        return final_lr + 0.5 * (max_lr - final_lr) * (1 + math.cos(theta))

    return lr_lambda


def create_cosine_lr_scheduler(  # noqa: PLR0913
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    initial_lr: float,
    max_lr: float,
    final_lr: float,
) -> LambdaLR:
    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=cosine_scheduler_with_warmup(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            initial_lr=initial_lr,
            max_lr=max_lr,
            final_lr=final_lr,
        ),
    )
