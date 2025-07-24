# 2025/7/19
# zhangzhong

from xihe.model import Transformer
from xihe.settings import ModelConfig, load_config
from torch.utils.data import DataLoader

# use pytorch lambdaLR to impl custom learning rate scheduler
from torch.optim import AdamW  # use this optimizer

# https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
from torch.optim.lr_scheduler import LambdaLR
import torch
import math
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm
from wandb.sdk.wandb_run import Run


# lr_scheduler = LambdaLR(
#     optimizer=AdamW(...),
#     lr_lambda=cosine_scheduler_with_warmup(
#         warmup_steps=2000,
#         total_steps=200000,
#         initial_lr=5e-5,
#         max_lr=1e-4,
#         final_lr=1e-5,
#     ),
# )


# TODO: 更名为 CausalLLMTrainer or GPTTrainer
class TransformerTrainer:
    def __init__(
        self,
        vocab_size: int,
        model: Transformer,
        # settings: ModelConfig,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        dataloader: DataLoader,
        max_norm: float = 1.0,
        run: Run | None = None,
        device="cuda",
        # dtype: str = "float32", 我觉得先不用这个参数吧
        # 其实混合精度的原理我还不理解
        # 而且我还需要弄清楚一件事情
        # 如果optimizer之前绑定了模型的参数
        # 然后模型又to cuda了，这样写是不是不对的？
        # 果然是不对的，
    ):
        self.vocab_size = vocab_size
        # self.config = settings
        # Initialize model, tokenizer, optimizer, etc. based on the config
        # 模型需要在外部to device上
        # self.model = model.to(device)
        self.model = model
        # self.tokenizer = model.tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = device
        self.run = run
        self.max_norm = max_norm
        self.loss_fn = torch.nn.CrossEntropyLoss()

    # TODO: 我们应该先把trainer给跑起来，在上这些东西
    # https://docs.pytorch.org/docs/stable/amp.html
    def train(self):
        # torch.amp.Scaler 需要需要保存状态呢？
        # 还真需要保存啊
        # 这里如果使用了DDP，应该是需要这是为对应的rank的吧
        scaler = torch.amp.GradScaler("cuda")

        self.model.train()
        for step, batch in tqdm(enumerate(self.dataloader)):
            self.optimizer.zero_grad()

            # 因为咱也没有支持bfloat16的设置，所以就不进行配置了
            inputs = batch["input_ids"].to(self.device)
            inputs = inputs.to(self.device)
            labels = inputs.clone()  # For simplicity, using inputs as labels
            shifted_labels = labels[:, 1:].contiguous()
            # labels.shape = [batch_size, seq_length]
            # labels = batch["labels"].to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(inputs)
                # logits.shape = [batch_size, seq_length, vocab_size]
                # 要在这里构造labels
                # 要使用cross entropy loss
                # 需要手动将logits和labels shift一下
                # logits的最后一个不要，因为那是整个序列的预测 我们没有他的golden
                # labels的第一个不要，因为没有对应的logits预测
                shifted_logits = logits[:, :-1, :].contiguous()

                # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                loss: Tensor = self.loss_fn(
                    shifted_logits.view(-1, self.vocab_size),
                    shifted_labels.view(-1),
                )
            if self.run:
                # Log loss to wandb
                self.run.log({"loss": loss.item()})
                # log lr
                self.run.log({"learning_rate": self.scheduler.get_last_lr()[0]})

            # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # 我们在这里需要unscale 因为需要做clip
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(self.optimizer)
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
            scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            # scaler need to update its scaling factor, because of dynamic scaling tricks
            scaler.update()

            # Update the learning rate
            self.scheduler.step()

            # if step % self.config.logging_steps == 0:
            #     print(f"Step {step}, Loss: {loss.item()}")

            # if step % self.config.save_steps == 0:
            #     # Save model checkpoint
            #     self.save_model()

    # TODO: 先不急，等SFT训练完了之后，在eval吧
    def evaluate(self):
        # Implement the evaluation logic
        pass

    def save_model(self):
        # Save the model and tokenizer to the specified paths
        # TODO: 如果训练中断，想要重启训练，是不是还得保存optimizer scheduelr dataloader的状态
        # torch.save(self.model.state_dict(), self.config.checkpoint_path)
        pass

    def load_model(self):
        # Load the model and tokenizer from the specified paths
        pass
