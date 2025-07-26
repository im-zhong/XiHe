# 2025/7/24
# zhangzhong


from pathlib import Path
import wandb
from wandb.sdk.wandb_run import Run
from typing import Any

from torch import nn
import torch
from xihe.settings import Config
import os

# 需要保存的东西
# model
# optimizer
# gradscaler
# lr_scheduler
# epoch/step
# config.toml
# seed 这个应该放在配置文件里面
# 随机数的状态
# dataloader也需要保存，不过要怎么保存呢 https://huggingface.co/docs/datasets/stream#save-a-dataset-checkpoint-and-resume-iteration
# wandb

# TODO: ddp下多个进程需要保存的东西一样吗？需要每个进程保存自己的optimiezr,scaler吗？
# 只有dataloader需要每个进程保存一份


# 这个东西应该是可以单独测试的
# 不应该依赖任何别的东西
# 就是一个单纯的读取state dict的类而已
class Checkpoint:
    # 为了强制保存所有应该保存的东西
    # 这里应该把所有应该保存的东西都列出来
    # 然后必须要保证名字和dict里面的名字是一样的
    def __init__(
        self,
        config: Config,
        model: dict[str, Any],
        optimizer: dict[str, Any],
        scheduler: dict[str, Any],
        grad_scaler: dict[str, Any],
        step: int,
        dataloader: list[Any],
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.step = step
        self.dataloader = dataloader

    def save(self, path: Path):
        os.makedirs(path.parent, exist_ok=True)
        torch.save(self.get_state_dict(), path)

    # 只需要定义一组key即可
    def get_state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.get_model_state_dict(),
            "optimizer": self.get_optimizer_state_dict(),
            "scheduler": self.get_scheduler_state_dict(),
            "dataloader": self.dataloader,  # Assuming single rank for now
            "grad_scaler": self.get_grad_scaler_state_dict(),
            "config": self.get_config(),
        }

    def get_config(self) -> Config:
        return self.config

    def get_model_state_dict(self) -> dict[str, Any]:
        return self.model

    def get_optimizer_state_dict(self) -> dict[str, Any]:
        return self.optimizer

    def get_scheduler_state_dict(self) -> dict[str, Any]:
        return self.scheduler

    def get_dataloader_state_dict(self, rank: int) -> dict[str, Any]:
        return self.dataloader[rank]

    def get_grad_scaler_state_dict(self) -> dict[str, Any]:
        return self.grad_scaler

    def get_step(self) -> int:
        return self.step


# 还得再写一个工厂函数
# 就是load_checkpoint
# 然后他会构造一个checkpoint对象 这就ok了！
def load_ckpt_from_path(path: Path) -> Checkpoint:
    checkpoint = torch.load(path, weights_only=False)
    return Checkpoint(
        config=checkpoint["config"],
        model=checkpoint["model"],
        optimizer=checkpoint["optimizer"],
        scheduler=checkpoint["scheduler"],
        grad_scaler=checkpoint["grad_scaler"],
        step=checkpoint["step"],
        dataloader=checkpoint["dataloader"],
    )
