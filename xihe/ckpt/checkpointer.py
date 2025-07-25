# 2025/7/24
# zhangzhong
# ~1500

from xihe.trainer import TransformerTrainer
from xihe.settings import load_config
from xihe.dataset import create_dataset, PackingDataset
from xihe.trainer.optimizer import create_optimizer, create_cosine_lr_scheduler
from pathlib import Path
import wandb
from wandb.sdk.wandb_run import Run
from torchdata.stateful_dataloader import StatefulDataLoader
from typing import Any

from torch import nn
import torch

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


class Checkpoint:
    def __init__(self, run: Run, model: Transformer, optimizer, scheduler):
        self.run = run
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    # https://docs.wandb.ai/guides/runs/resuming/
    def load_wandb_run(
        self,
        id: str,
        entity: str | None = None,
        project: str | None = None,
    ) -> Run:
        # 这个entity应该是你登陆之后自己检测的
        # 所以不需要传入的
        return wandb.init(entity=entity, project=project, id=id, resume="must")

    def load_model(self, model: nn.Module, state_dict: dict[str, Any]) -> nn.Module:
        # 想要load就得先创建模型
        # 但是checkpoint不应该负责创建模型
        model.load_state_dict(state_dict)
        return model

    def save(self, path: Path):
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(state, path)
        self.run.save(path)

    def load(self, path: Path):
        state = torch.load(path)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.run.restore(path)
