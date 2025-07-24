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


class Checkpointer:
    def __init__(self, run: Run, model: Transformer, optimizer, scheduler):
        self.run = run
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

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
