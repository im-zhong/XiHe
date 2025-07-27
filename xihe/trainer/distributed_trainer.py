# 2025/7/24
# zhangzhong
#

from xihe.model import Transformer

# https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch.optim import Optimizer
from tqdm import tqdm
from wandb.sdk.wandb_run import Run
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdata.stateful_dataloader import StatefulDataLoader
from xihe.optimizer import (
    create_optimizer,
    create_cosine_lr_scheduler,
)
from xihe.utils.wandb import init_wandb_run
from .basic_trainer import BasicGPTTrainer
from xihe.settings import Config
from xihe.tokenizer import create_tokenizer
from xihe.dataset import create_dataloader
import torch.distributed as dist
from typing import Any
from xihe.defs import defs

# 我感觉也没必要整两个，因为我们就是从Config里面读出来的
# 他们俩用一个就行了
# config用schema就行了
from xihe.ckpt import Checkpoint
from torch.amp.grad_scaler import GradScaler


# TODO: 更名为 CausalLLMTrainer or GPTTrainer
# 把rank放在参数里面吧
# 还有所有需要用的实例
# TODO: 最关键的问题是怎么协调train from scratch和train from checkpoint
#
class DistributedGPTTrainer:
    def __init__(
        self,
        rank: int,
        world_size: int,
        config: Config,
        run: Run | None = None,
        # dtype: str = "float32", 我觉得先不用这个参数吧
        # 其实混合精度的原理我还不理解
        # 而且我还需要弄清楚一件事情
        # 如果optimizer之前绑定了模型的参数
        # 然后模型又to cuda了，这样写是不是不对的？
        # 果然是不对的，
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.run = run
        self.config = config

    # def get_state_dict(self, step: int):
    #     # only rank 0 should save the checkpoint
    #     # save the model, optimizer, scheduler, scaler, and config
    #     # save the model state dict

    #     # 首先gather所有的dataloader state
    #     gathered_dataloader_state = None
    #     if self.rank == 0:
    #         gathered_dataloader_state = [None for _ in range(self.world_size)]
    #     dataloader_state = self.dataloader.state_dict()
    #     dist.gather_object(
    #         obj=dataloader_state, object_gather_list=gathered_dataloader_state, dst=0
    #     )

    #     if self.rank != 0:
    #         return

    #     checkpoint = {
    #         "model": self.model.state_dict(),
    #         "optimizer": self.optimizer.state_dict(),
    #         "scheduler": self.scheduler.state_dict(),
    #         "grad_scaler": self.grad_scaler.state_dict(),
    #         "dataloader": gathered_dataloader_state,
    #         # "config": self.config.model_dump(),
    #         "step": step,
    #         # TODO: maybe save the random state as well
    #         # 保存随机数状态是为了可复现
    #         # 但是我们的系统有什么复现的必要吗
    #         # 没必要，所以不保存了
    #         "rngom_state": get_all_rng_states(),
    #     }
    #     return checkpoint
    #     # save the checkpoint to a file
    #     # 应该是不需要在配置里面写上路径的
    #     # 需要在在.cache/checkpoint_1000.pt ?
    #     # 那如果是多次训练呢？
    #     # torch.save(checkpoint, self.config.checkpoint_path)

    # 感觉这里传入config对象更好
    # def train_from_scratch(rank: int, world_size: int, conf: Config):
    #     pass
    # 我有一个折衷的方案：就是把checkpoint作为可选参数放进来
    # 这样可以减少冗余的代码，这个函数就可以叫做train
    # 然后有一个内部函数叫做 train_loop 目前看来是比较合适的了
    def train(self, config: Config, ckpt: Checkpoint | None = None):
        if ckpt and config != ckpt.get_config():
            raise ValueError(
                "The provided checkpoint configuration does not match the current configuration."
            )

        rank = self.rank
        world_size = self.world_size
        self.setup(self.rank, self.world_size)
        # TODO: tmp set this code, if we impl ddp, should delete this
        # ? 这设置的不是torch 创建tensor的默认设备？
        # 这个并不是用来设置默认设备的，pytorch没有提供这个功能
        # 这个是设置默认的cuda设备的，也就是当你指定 cuda 但是没有指定 cuda:id 的时候的默认id
        torch.cuda.set_device(self.rank)  # Set the default CUDA device to 0

        # Load configuration
        # config = load_config(Path(args.conf))
        # 在这里创建wandb更好
        # 不对，其实wandb有智能检测的，只有是一个存在过的id
        # 你再次初始化wandb 其实也是resume
        # 所以没必要整两套逻辑
        run: Run = init_wandb_run(config=config)

        # get tokenizer from tokenizer configs
        # 但是我们不能直接依赖TokenizerConfig这个类
        # 要写一个函数来根据配置获取tokenizer
        # tokenizer = get_tokenizer(config.tokenizer)
        tokenizer = create_tokenizer(config.tokenizer.tokenizer_name)
        dataloader: StatefulDataLoader = create_dataloader(
            tokenizer=tokenizer,
            rank=self.rank,
            world_size=world_size,
            batch_size=config.trainer.batch_size,
            context_length=config.model.context_length,
            # TODO: 合并DatasetArgs和DatasetConfig
            datasets_args=config.dataloader.datasets,
            sampling_probabilities=config.dataloader.sampling_probabilities,
        )
        if ckpt:
            dataloader.load_state_dict(ckpt.get_dataloader_state_dict(self.rank))

        # # Initialize model
        # TODO: vocab size应该是不用设置的
        # 应该可以通过tokenizer对象来获取才对
        local_model = Transformer(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=config.model.context_length,
            num_layers=config.model.num_layers,
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_heads,
            intermediate_size=config.model.intermediate_size,
            device=config.trainer.device,  # Pass the device from config
        )
        if ckpt:
            local_model.load_state_dict(ckpt.get_model_state_dict())
        local_model = local_model.to(config.trainer.device)
        # TODO: need to call init_process_group, but where?
        model = DDP(module=local_model, device_ids=[rank])

        # 这里需要创建optimizer和scheduler
        # TODO: 这些东西都得重写，optimizer必须在模型的参数放在正确的设备上才能创建
        optimizer: Optimizer = create_optimizer(
            name=config.optimizer.optimizer_name,
            learning_rate=config.optimizer.initial_lr,
            weight_decay=config.optimizer.weight_decay,
            parameters=model.parameters(),
        )
        if ckpt:
            optimizer.load_state_dict(ckpt.get_optimizer_state_dict())

        lr_scheduler: LambdaLR = create_cosine_lr_scheduler(
            optimizer=optimizer,
            warmup_steps=config.trainer.warmup_steps,
            total_steps=config.trainer.total_steps,
            initial_lr=config.optimizer.initial_lr,
            max_lr=config.optimizer.max_lr,
            final_lr=config.optimizer.final_lr,
        )
        if ckpt:
            lr_scheduler.load_state_dict(ckpt.get_scheduler_state_dict())

        grad_scaler = GradScaler(
            "cuda", enabled=config.model.mixed_precision
        )  # Enable mixed precision training
        if ckpt:
            grad_scaler.load_state_dict(ckpt.get_grad_scaler_state_dict())

        # # Initialize the trainer
        trainer = BasicGPTTrainer(
            rank=rank,
            world_size=world_size,
            grad_scaler=grad_scaler,
            vocab_size=tokenizer.vocab_size,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            # dataloader=dataloader,
            device=config.trainer.device,
            # dtype=config.model.dtype,
            run=run,
        )

        start_step = ckpt.get_step() if ckpt else 0
        # Start training
        self.train_loop(
            start_step=start_step,
            model=model,
            dataloader=dataloader,
            trainer=trainer,
            rank=rank,
            scheduler=lr_scheduler,
            world_size=world_size,
        )

        self.cleanup()

    # only work with nvidia GPUs
    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.cuda.set_device(rank)
        # Setup distributed training environment
        torch.distributed.init_process_group(
            # backend="nccl",
            rank=rank,
            world_size=world_size,
        )

    def cleanup(self):
        torch.distributed.destroy_process_group()

    # 有一种让Trainer完全剥离ckpter的方法
    # 就是把这个train函数写到外面 写到main文件里面
    # 好像这样就是对的，这个类就不是DistributedTrainer了
    # 他反而是对distributed无感的！
    # 我们就可以在外部同时实现单进程和多进程的训练！
    # 单进程的训练就不用实现了，但是需要进行测试
    # 多进程的训练写在distributed_pretrain.py里面就可以了
    # TODO: 我们应该先把trainer给跑起来，在上这些东西
    # https://docs.pytorch.org/docs/stable/amp.html
    def train_loop(
        self,
        start_step: int,
        model: DDP,
        dataloader: StatefulDataLoader,
        trainer: BasicGPTTrainer,
        scheduler: LambdaLR,
        rank: int,
        world_size: int,
    ):
        # get trainer from

        # torch.amp.Scaler 需要需要保存状态呢？
        # 还真需要保存啊
        # 这里如果使用了DDP，应该是需要这是为对应的rank的吧
        # scaler = torch.amp.GradScaler("cuda")

        # TODO: 这个train函数无法承担任何构造对象的过程
        # 他只能直接从外部获取对象，然后使用
        # 否则创建的过程就会和训练的过程耦合
        # 现在无法实现checkpoint机制了
        # 也就是这里的scaler和model的创建都要移动到train函数外面

        # TODO: 我记得DDP加载模型还需要barrier来着

        model.train()

        for step, batch in tqdm(enumerate(dataloader, start=start_step)):
            loss = trainer.train_step(step=step, batch=batch)

            # 因为会有多个进程，并且gradient是在backward之后才能汇总起来
            # 所以loss只能按照rank来打印了 没法打印整体的
            if self.run:
                # Log loss to wandb
                self.run.log({f"loss-{self.rank}": loss})
                # log lr
                # 但是learning rate所有的进程都是一样的
                if self.rank == 0:
                    self.run.log({"learning_rate": scheduler.get_last_lr()[0]})

            self.save_checkpoint(
                step=step,
                trainer=trainer,
                dataloader=dataloader,
            )

            # 在这里实现save checkpoint
            # 因为我们要保存config对象，所以可以把config传进来
            # TODO: 如果传进来的话，所有训练需要的对象的构造就都可以在trainer里面完成
            # 但是这样做并不合适啊
            # 先写吧，整体的架构肯定需要重构一下的

            # if step % 1000 == 0:
            #     self.save_checkpoint(step=step, trainer=trainer, dataloader=dataloader)

    # 算了，收集dataloader state extract a function
    def get_dataloader_state(
        self, dataloader: StatefulDataLoader, rank: int
    ) -> list[Any] | None:
        # prepare local data
        dataloader_state = dataloader.state_dict()

        # rank 0 collect all states
        # and other ranks do not
        gathered_dataloader_state = None
        if self.rank == 0:
            # prepare a list to gather all states
            gathered_dataloader_state = [None for _ in range(self.world_size)]

        dist.gather_object(
            obj=dataloader_state,
            object_gather_list=gathered_dataloader_state,
            dst=0,
        )
        return gathered_dataloader_state

    def save_checkpoint(
        self, step: int, trainer: BasicGPTTrainer, dataloader: StatefulDataLoader
    ):
        # only rank 0 should save the checkpoint
        # 先从其他节点哪里拿到dataloader
        gathered_dataloader_state = self.get_dataloader_state(
            dataloader=dataloader, rank=self.rank
        )

        if self.rank != 0:
            return

        assert gathered_dataloader_state is not None, (
            "Dataloader state should not be None"
        )
        checkpoint = Checkpoint(
            config=self.config,
            **trainer.get_state_dict(step=step),
            dataloader=gathered_dataloader_state,
        )
        checkpoint.save(
            path=defs.get_ckpt_path(project=self.config.wandb.project, step=step)
        )

        # config = checkpoint.load_config()

        # save the checkpoint to a file
        # 应该是不需要在配置里面写上路径的
        # 需要在在.cache/checkpoint_1000.pt ?
        # 那如果是多次训练呢？
        # torch.save(
        #     checkpoint,
        #     defs.get_ckpt_path(project=self.config.wandb.project, step=step),
        # )
