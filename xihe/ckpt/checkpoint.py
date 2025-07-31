# 2025/7/24
# zhangzhong


import bisect
from pathlib import Path
from typing import Any

import torch

from xihe.defs import defs
from xihe.settings import Config

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


# 需要重新写一个CheckpointPath


class CheckpointPath:
    @property
    def ckpts_dir(self) -> Path:
        return defs.cache_dir / "ckpts"

    def get_ckpts_dir(self, project: str) -> Path:
        project_dir = self.ckpts_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def find_step_ckpts(self, project_dir: Path) -> list[Path]:
        # project_dir = self.get_ckpts_dir(project)
        return sorted(
            project_dir.glob("ckpt_step*.tar"), key=lambda x: int(x.stem.split("_")[-1])
        )

    def find_best_ckpt(self, project_dir: Path) -> list[Path]:
        # project_dir = self.get_ckpts_dir(project)
        # 哦！！！不对，这里的排序是不对的！
        # 我们需要根据step来排序，不能根据字典序来排序！

        return sorted(
            project_dir.glob("best_ckpt_step*.tar"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )

    def get_step_ckpt_path(self, project_dir: Path, step: int) -> Path:
        # project_dir = self.get_ckpts_dir(project)
        return project_dir / f"ckpt_step_{step}.tar"

    def get_best_ckpt_path(self, project_dir: Path, step: int) -> Path:
        # project_dir = self.get_ckpts_dir(project)
        return project_dir / f"best_ckpt_step_{step}.tar"


ckpt_defs = CheckpointPath()


# 这个东西应该是可以单独测试的
# 不应该依赖任何别的东西
# 就是一个单纯的读取state dict的类而已
class Checkpoint:
    # 为了强制保存所有应该保存的东西
    # 这里应该把所有应该保存的东西都列出来
    # 然后必须要保证名字和dict里面的名字是一样的
    def __init__(  # noqa: PLR0913
        self,
        config: Config,
        model: dict[str, Any],
        optimizer: dict[str, Any],
        scheduler: dict[str, Any],
        grad_scaler: dict[str, Any],
        step: int,
        dataloader: list[Any],
        loss: float,
        num_tokens: int,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.step = step
        self.dataloader = dataloader
        self.loss = loss
        self.num_tokens = num_tokens
        # 不对啊，这是保存一个文件的
        # 我们需要管理的是一个目录 dir
        # 所以我们最好是基于这个类再写一个新的类
        # self.best_loss = float("inf")
        # self.current_saved_steps: list[int] = []

    def save(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.get_state_dict(), path)

    # def get_best_loss(self) -> float:
    #     return self.best_loss

    # def get_current_saved_steps(self) -> list[int]:

    def get_loss(self) -> float:
        return self.loss

    def get_num_tokens(self) -> int:
        return self.num_tokens

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
            "loss": self.get_loss(),
            "num_tokens": self.get_num_tokens(),
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
def load_ckpt_from_path(path: Path) -> Checkpoint | None:
    # tmd可能没有这个文件啊
    if not path.exists():
        return None
    checkpoint = torch.load(path, weights_only=False)
    return Checkpoint(
        config=checkpoint["config"],
        model=checkpoint["model"],
        optimizer=checkpoint["optimizer"],
        scheduler=checkpoint["scheduler"],
        grad_scaler=checkpoint["grad_scaler"],
        step=checkpoint["step"],
        dataloader=checkpoint["dataloader"],
        loss=checkpoint["loss"],
        num_tokens=checkpoint["num_tokens"],
    )


class CheckpointManager:
    # 这个东西比我想的要复杂一些
    # 要不回来再写吧
    # 一般我们想要用一个checkpoint来训练的时候
    # 我们希望的是一个全新的训练过程
    # 所以在一个文件夹里面，除了当前的文件，不应该有其他的权重文件
    # 否则就会被覆盖掉
    # 所以可以做一个检测
    # 这样我们实现起来也比较简单，不需要考虑特别多的复杂的情况
    def __init__(self, keep_num: int, save_steps: int, ckpt_dir: Path) -> None:
        # 实际上只用了 keep num 和 save steps
        # 那么我们把config换成这两个参数就好了
        self.keep_num = keep_num
        self.save_steps = save_steps
        # read all the checkpoint files in this directory
        self.ckpt_dir = ckpt_dir

        # 既然我们不需要维护checkpoint对象，也就不需要这些代码了
        # self.checkpoints: list[Checkpoint] = []
        # for file in ckpt_dir.glob("ckpt_step_*.tar"):
        #     # self.add_checkpoint(load_ckpt_from_path(file))
        #     ckpt = load_ckpt_from_path(file)
        #     if ckpt is not None:
        #         self.checkpoints.append(ckpt)
        # # 将checkpoints中的checkpoint按照step排序
        # self.checkpoints.sort(key=lambda x: x.get_step())
        # # 根据config.checkpoint.keep_num来决定保留多少个checkpoint
        # if len(self.checkpoints) > self.config.checkpoint.keep_num:
        #     self.checkpoints = self.checkpoints[-self.config.checkpoint.keep_num :]
        # self.max_step = self.checkpoints[-1].get_step() if self.checkpoints else -1

        # 这个东西的名字应该写在defs里面
        self.best_loss = float("inf")
        best_ckpt_path = ckpt_defs.find_best_ckpt(self.ckpt_dir)
        # 应该是只有一个的
        # 但是就算有多个也没关系
        # 我们就选择最后一个就行了
        # 其他的都删掉
        if len(best_ckpt_path) > 0:
            best_ckpt_path = best_ckpt_path[-1]
            best_ckpt: Checkpoint | None = load_ckpt_from_path(best_ckpt_path)
            if best_ckpt is not None:
                self.best_loss = best_ckpt.get_loss()

        # 这个逻辑还挺绕的
        # best checkpoint需要和step interval checkpoint分开
        # 如果要放在一起保存的话，就会出现多个best的情况

    def clear_checkpoints(self) -> None:
        # 清空所有的checkpoint文件
        # 就把ckpt_dir目录下的所有文件都删除掉
        for file in self.ckpt_dir.iterdir():
            if file.is_file():
                file.unlink(missing_ok=True)

        # 重新初始化best_loss
        self.best_loss = float("inf")

    def get_best_loss(self) -> float:
        return self.best_loss

    # 这里要保存的step 不一定比最大的step大
    # 所以还是需要排序的
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        # 这里的step是当前的step
        # 需要保存的checkpoint是当前的checkpoint
        # 需要保存到ckpt_dir目录下
        # 需要保证文件名是唯一的
        # 如果已经存在了，就覆盖掉
        # filename = self.ckpt_dir / f"ckpt_step_{step}.tar"
        # checkpoint.save(filename)

        # TODO: 写一个函数，从目录中读取所有的checkpoint
        # 并删除多余的checkpoint
        # self.checkpoints.append(checkpoint)
        # self.checkpoints.sort(key=lambda x: x.get_step())
        # self.max_step = max(self.max_step, step)
        # if len(self.checkpoints) > self.config.checkpoint.keep_num:
        #     # 还要在文件系统中删掉啊
        #     for ckpt in self.checkpoints[: -self.config.checkpoint.keep_num]:
        #         (self.ckpt_dir / f"ckpt_step_{ckpt.get_step()}.tar").unlink(
        #             missing_ok=True
        #         )
        #     self.checkpoints = self.checkpoints[-self.config.checkpoint.keep_num :]

        step = checkpoint.get_step()
        if self.need_save_interval(step):
            self.save_step_checkpoint(step, checkpoint)

        loss = checkpoint.get_loss()
        if self.need_save_best(loss):
            self.save_best_checkpoint(checkpoint)

    def save_step_checkpoint(self, step: int, checkpoint: Checkpoint) -> None:
        # filename = self.ckpt_dir / f"ckpt_step_{step}.tar"
        filename = ckpt_defs.get_step_ckpt_path(self.ckpt_dir, step)
        checkpoint.save(filename)

        # 在这里处理删除多余checkpoint的逻辑
        self.delete_old_checkpoints(step=step, ckpt_dir=self.ckpt_dir)

    def update_best_loss(self, loss: float) -> None:
        # 更新best loss
        # 这个函数是为了在训练过程中更新best loss的
        self.best_loss = min(self.best_loss, loss)

    # 我怎么知道best loss呢？
    # 所以checkpoint里面还需要保存loss 。。。
    def save_best_checkpoint(self, checkpoint: Checkpoint) -> None:
        # 这里的checkpoint是当前的checkpoint
        # 需要保存到ckpt_dir目录下
        # 需要保证文件名是唯一的
        # 如果已经存在了，就覆盖掉
        # 如果loss相同，但是因为step更新，所以还是要保存

        # if self.best_ckpt is None:
        #     self.best_ckpt = checkpoint

        # if checkpoint.get_loss() > self.best_ckpt.get_loss():
        #     return
        if checkpoint.get_loss() > self.best_loss:
            return

        # 哎！我好像知道问题了，这个best loss每个进程都是需要更新的！
        # self.best_loss = checkpoint.get_loss()
        self.update_best_loss(checkpoint.get_loss())
        # filename = self.ckpt_dir / "ckpt_best.tar"
        filename = ckpt_defs.get_best_ckpt_path(
            project_dir=self.ckpt_dir, step=checkpoint.get_step()
        )
        checkpoint.save(filename)
        # self.best_ckpt = checkpoint

        # 这里也需要把旧的 best ckpt给删掉啊
        # 只需要删除旧的best ckpt就行了
        best_ckpt_path = ckpt_defs.find_best_ckpt(self.ckpt_dir)
        if len(best_ckpt_path) > 1:
            # 删除除了最新的best ckpt之外的所有best ckpt
            for path in best_ckpt_path[:-1]:
                path.unlink(missing_ok=True)

    def load_checkpoints(self, ckpt_dir: Path) -> list[Checkpoint]:
        checkpoint_pathes = ckpt_defs.find_step_ckpts(project_dir=ckpt_dir)
        # 既然我们不需要维护checkpoint对象，也就不需要这些代码了
        checkpoints: list[Checkpoint] = []
        for file in checkpoint_pathes:
            # self.add_checkpoint(load_ckpt_from_path(file))
            ckpt = load_ckpt_from_path(file)
            if ckpt is not None:
                checkpoints.append(ckpt)
        return checkpoints

    def delete_old_checkpoints(self, step: int, ckpt_dir: Path) -> None:
        checkpoints: list[Checkpoint] = self.load_checkpoints(ckpt_dir)

        # 只删除step之前的超过配置数量的checkpoint吧
        # 将checkpoints中的checkpoint按照step排序
        checkpoints.sort(key=lambda x: x.get_step())
        # The return value i is such that all e in a[:i] have e <= x, and all e in a[i:] have e > x. So if x already appears in the list, a.insert(i, x) will insert just after the rightmost x already there.
        index: int = bisect.bisect_right(checkpoints, step, key=lambda x: x.get_step())
        checkpoints = checkpoints[:index]

        # 根据config.checkpoint.keep_num来决定保留多少个checkpoint
        for ckpt in checkpoints[: -self.keep_num]:
            ckpt_defs.get_step_ckpt_path(
                project_dir=self.ckpt_dir, step=ckpt.get_step()
            ).unlink(missing_ok=True)
            # (self.ckpt_dir / f"ckpt_step_{ckpt.get_step()}.tar").unlink(missing_ok=True)

    # 最低loss和按照step interval保存的ckpt需不需要区分？
    # 肯定还是需要区分的
    def need_save(self, step: int, loss: float) -> bool:
        # 需要保存的step是当前的step
        # 如果当前的step大于最大的step，就需要保存
        # 否则就不需要保存
        return self.need_save_interval(step) or self.need_save_best(loss)

    # 所以这个need save实际上是由两个need组成的
    def need_save_interval(self, step: int) -> bool:
        # 需要保存的step是当前的step
        # 如果当前的step大于最大的step，就需要保存
        # 否则就不需要保存
        return step % self.save_steps == 0

    def need_save_best(self, loss: float) -> bool:
        # 需要保存的loss是当前的loss
        # 如果当前的loss小于等于最好的loss，就需要保存
        # 否则就不需要保存
        return loss <= self.best_loss
