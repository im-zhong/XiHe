# 2025/7/24
# zhangzhong
#

# https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
from typing import Any

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from wandb.sdk.wandb_run import Run
from xihe.model import Transformer


# TODO: 更名为 CausalLLMTrainer or GPTTrainer
# 把rank放在参数里面吧
# 还有所有需要用的实例
class BasicGPTTrainer:
    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        model: Transformer | DistributedDataParallel,
        # settings: ModelConfig,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        # dataloader: StatefulDataLoader,
        grad_scaler: GradScaler,
        # rank: int,
        # world_size: int,
        max_norm: float = 1.0,
        run: Run | None = None,
        device: str = "cuda",
        # dtype: str = "float32", 我觉得先不用这个参数吧
        # 其实混合精度的原理我还不理解
        # 而且我还需要弄清楚一件事情
        # 如果optimizer之前绑定了模型的参数
        # 然后模型又to cuda了，这样写是不是不对的？
        # 果然是不对的，
    ) -> None:
        self.vocab_size = vocab_size
        # self.config = settings
        # Initialize model, tokenizer, optimizer, etc. based on the config
        # 模型需要在外部to device上
        # self.model = model.to(device)
        self.model = model
        # self.tokenizer = model.tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.dataloader = dataloader
        self.device = device
        self.run = run
        self.max_norm = max_norm
        self.grad_scaler = grad_scaler
        # self.rank = rank
        # self.world_size = world_size
        self.loss_fn = torch.nn.CrossEntropyLoss()

    # 咱们写两个版本的trainer 看看显存占用吧

    def train_step(self, batch: dict[str, Tensor]) -> float:
        self.optimizer.zero_grad()

        # 为什么这里运行第二次的显存会增加？
        # 是不是代码有些地方写的效率比较低？怎么可以优化一下？

        # 因为咱也没有支持bfloat16的设置，所以就不进行配置了
        inputs = batch["input_ids"].to(self.device)
        labels = inputs.clone()  # For simplicity, using inputs as labels
        labels = labels[:, 1:].contiguous()
        # labels.shape = [batch_size, seq_length]
        # labels = batch["labels"].to(self.device)
        # self.grad_scaler.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.grad_scaler.is_enabled(),
        ):
            logits = self.model(inputs)
            # logits.shape = [batch_size, seq_length, vocab_size]
            # 要在这里构造labels
            # 要使用cross entropy loss
            # 需要手动将logits和labels shift一下
            # logits的最后一个不要，因为那是整个序列的预测 我们没有他的golden
            # labels的第一个不要，因为没有对应的logits预测
            logits = logits[:, :-1, :].contiguous()

            # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            loss: Tensor = self.loss_fn(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
            )

        # loss_item = float(loss.item())

        # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
        self.grad_scaler.scale(loss).backward()

        # 我们在这里需要unscale 因为需要做clip
        # Unscales the gradients of optimizer's assigned params in-place
        self.grad_scaler.unscale_(self.optimizer)

        # Clip gradients to avoid exploding gradients
        # torch.nn.utils.clip_grad_norm_(
        #     parameters=self.model.parameters(), max_norm=self.config.grad_clip
        # )
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        # ``scaler.step()`` first unscales the gradients of the optimizer's assigned parameters.
        # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        # self.optimizer.step()
        self.grad_scaler.step(self.optimizer)

        # Updates the scale for next iteration.
        # scaler need to update its scaling factor, because of dynamic scaling tricks
        self.grad_scaler.update()

        # Update the learning rate
        self.scheduler.step()

        # 试一下清空缓存
        # 作用不大，而且速度变慢了
        # torch.cuda.empty_cache()

        # if step % self.config.logging_steps == 0:
        #     print(f"Step {step}, Loss: {loss.item()}")

        # if step % self.config.save_steps == 0:
        #     # Save model checkpoint
        #     self.save_model()
        return 100

    def get_model_state_dict(self) -> dict[str, Any]:
        match self.model:
            case DistributedDataParallel():
                # If model is wrapped in DDP, we need to access the underlying model
                return self.model.module.state_dict()
            case Transformer():
                # If model is a Transformer, we can directly return its state dict
                return self.model.state_dict()
            case _:
                msg = "Unsupported model type for state dict retrieval."
                raise ValueError(msg)

    def get_state_dict(self, step: int) -> dict[str, Any]:
        # only rank 0 should save the checkpoint
        # save the model, optimizer, scheduler, scaler, and config
        # save the model state dict

        # 首先gather所有的dataloader state
        # gathered_dataloader_state = None
        # if self.rank == 0:
        #     gathered_dataloader_state = [None for _ in range(self.world_size)]
        # dataloader_state = self.dataloader.state_dict()
        # dist.gather_object(
        #     obj=dataloader_state, object_gather_list=gathered_dataloader_state, dst=0
        # )

        return {
            # 保存时如果是在 DDP 环境中，最好用 model.module.state_dict() 保存（而不是直接 model.state_dict()），这样就不会带 "module." 前缀。
            "model": self.get_model_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "grad_scaler": self.grad_scaler.state_dict(),
            # "dataloader": gathered_dataloader_state,
            # "config": self.config.model_dump(),
            "step": step,
            # TODO: maybe save the random state as well
            # 保存随机数状态是为了可复现
            # 但是我们的系统有什么复现的必要吗
            # 没必要，所以不保存了
            # "rngom_state": get_all_rng_states(),
        }
        # return checkpoint
        # save the checkpoint to a file
        # 应该是不需要在配置里面写上路径的
        # 需要在在.cache/checkpoint_1000.pt ?
        # 那如果是多次训练呢？
        # torch.save(checkpoint, self.config.checkpoint_path)
