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
from wandb.sdk.wandb_run import Run
from xihe.settings import Config, WandbConfig
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.optim import Optimizer, Adam, AdamW
from torch.utils.data import DataLoader
from xihe.settings import Config
from torchdata.stateful_dataloader import StatefulDataLoader
from xihe.tokenizer import create_tokenizer
from xihe.dataset import create_dataloader
import torch.distributed as dist

# 我感觉也没必要整两个，因为我们就是从Config里面读出来的
# 他们俩用一个就行了
# config用schema就行了
from xihe.schemas import DatasetArgs


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
    def train_from_scratch(self, rank, world_size, config: Config):
        # TODO: tmp set this code, if we impl ddp, should delete this
        # ? 这设置的不是torch 创建tensor的默认设备？
        # 这个并不是用来设置默认设备的，pytorch没有提供这个功能
        # 这个是设置默认的cuda设备的，也就是当你指定 cuda 但是没有指定 cuda:id 的时候的默认id
        torch.cuda.set_device(rank)  # Set the default CUDA device to 0

        # Load configuration
        # config = load_config(Path(args.conf))
        # 在这里创建wandb更好
        run: Run = init_wandb_run(config=config)

        # get tokenizer from tokenizer configs
        # 但是我们不能直接依赖TokenizerConfig这个类
        # 要写一个函数来根据配置获取tokenizer
        # tokenizer = get_tokenizer(config.tokenizer)
        tokenizer = create_tokenizer(config.tokenizer.tokenizer_name)
        dataloader: StatefulDataLoader = create_dataloader(
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            batch_size=config.trainer.batch_size,
            context_length=config.model.context_length,
            # TODO: 合并DatasetArgs和DatasetConfig
            datasets_args=config.dataloader.datasets,
            sampling_probabilities=config.dataloader.sampling_probabilities,
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
        model = model.to(config.trainer.device)
        model = DDP(model, device_ids=[rank])

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

        grad_scaler = torch.amp.GradScaler("cuda")  # Enable mixed precision training

        # # Initialize the trainer
        trainer = BasicGPTTrainer(
            rank=rank,
            world_size=world_size,
            grad_scaler=grad_scaler,
            vocab_size=config.tokenizer.vocab_size,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            # dataloader=dataloader,
            device=config.trainer.device,
            # dtype=config.model.dtype,
            run=run,
        )

        # Start training
        self.train_loop(
            model=model,
            dataloader=dataloader,
            trainer=trainer,
            rank=rank,
            scheduler=lr_scheduler,
            world_size=world_size,
        )

    # 这个传入一个checkpoint对象更好吧
    # 咱们先不直接写checkpoint了，先用最简单的方式把这个函数写出来
    # 再看看能不能重构一个checkpoint出来
    # def train_from_checkpoint(rank: int, world_size: int, checkpoint: dict[str, Any]):
    #     torch.cuda.set_device(rank)  # Set the default CUDA device to 0
    #     wandb.login()

    #     set_all_rng_states(checkpoint["rng_state"])

    #     # 需要设计一种文件结构
    #     # 不对啊，其实没有任何的必要
    #     # 我们就全部放在一个大的dict里面就行了
    #     # 这样最灵活了，所以这应该是一个file而不是一个dir
    #     # checkpoint: dict[str, Any] = torch.load(ckpt_file)
    #     # resume config first
    #     # TODO: 这里的每一步都应该可以被测试
    #     config: Config = Config(**checkpoint["config"])

    #     # then load wandb
    #     run: Run = load_wnadb_run(wandb_config=config.wandb)

    #     # TODO: 这个还需要研究
    #     # load dataloader
    #     # 那就只需要构建dataset
    #     # 生成dataloader
    #     # 然后load state就行了

    #     # 算了，我一直纠结的点在于，我想要测试load state的过程
    #     # 但是这一部分也没什么好测的呀
    #     tokenizer = create_tokenizer(config.tokenizer.tokenizer_name)
    #     dataloader = get_dataloader(
    #         tokenizer=tokenizer,
    #         rank=rank,
    #         world_size=world_size,
    #         config=config,
    #     )
    #     # TODO: 这些特定的save和load逻辑都应该封装在一起
    #     # 方便阅读理解和测试
    #     dataloader.load_state_dict(checkpoint["dataloader"][rank])

    #     grad_scaler = torch.amp.GradScaler("cuda", enabled=config.trainer.use_amp)
    #     grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    #     # load model
    #     # define model first
    #     model = Transformer(
    #         vocab_size=config.tokenizer.vocab_size,
    #         max_seq_len=config.model.context_length,
    #         num_layers=config.model.num_layers,
    #         hidden_size=config.model.hidden_size,
    #         num_heads=config.model.num_heads,
    #         intermediate_size=config.model.intermediate_size,
    #         device=config.trainer.device,  # Pass the device from config
    #     )
    #     # model = checkpoint.load_model(model)
    #     model.load_state_dict(checkpoint["model"])
    #     model = model.to(rank)
    #     model = DDP(model, device_ids=[rank])

    #     # load optimizer
    #     optimizer: Optimizer = create_optimizer(
    #         name=config.optimizer.optimizer_name,
    #         learning_rate=config.optimizer.initial_lr,
    #         weight_decay=config.optimizer.weight_decay,
    #         parameters=model.parameters(),
    #     )
    #     optimizer.load_state_dict(checkpoint["optimizer"])

    #     # load scheduler
    #     lr_scheduler: LambdaLR = create_cosine_lr_scheduler(
    #         optimizer=optimizer,
    #         warmup_steps=config.trainer.warmup_steps,
    #         total_steps=config.trainer.total_steps,
    #         initial_lr=config.optimizer.initial_lr,
    #         max_lr=config.optimizer.max_lr,
    #         final_lr=config.optimizer.final_lr,
    #     )
    #     lr_scheduler.load_state_dict(checkpoint["scheduler"])

    #     # # Initialize the trainer
    #     # TODO: 这里感觉也不是很好
    #     # 模型是不是应该只加载一次？然后由DDP负责将模型复制到每个GPU上？
    #     # 现在的写法，每个rank都会创建一个新的模型实例，都会加载之前的权重
    #     # 然后DDP又复制了一次
    #     # 不过这个点感觉消耗的时间并不多，所以先这样吧
    #     trainer = BasicGPTTrainer(
    #         # config=config,
    #         vocab_size=config.tokenizer.vocab_size,
    #         model=model,
    #         optimizer=optimizer,
    #         scheduler=lr_scheduler,
    #         dataloader=dataloader,
    #         device=config.trainer.device,
    #         # dtype=config.model.dtype,
    #         grad_scaler=grad_scaler,
    #         rank=rank,
    #         world_size=world_size,
    #         run=run,
    #     )

    #     # Start training
    #     self.train(trainer)

    # TODO: 感觉把这个类写出来，可以做单元测试了呀
    # 这个类也要尽可能的包含功能，这样可以让main.py代码更少
    # def load_state_dict(self, rank: int, state_dict: dict[str, Any]):
    #     set_all_rng_states(state_dict["rng_state"])

    #     self.model.load_state_dict(state_dict["model"])
    #     self.model = self.model.to(rank)

    #     self.dataloader.load_state_dict(state_dict["dataloader"][rank])
    #     self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    #     # TODO: model = model(DDP) 这一步要放在哪里呢？
    #     self.optimizer.load_state_dict(state_dict["optimizer"])
    #     self.lr_scheduler.load_state_dict(state_dict["scheduler"])

    #     pass

    # only work with nvidia GPUs
    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.cuda.set_device(rank)
        # Setup distributed training environment
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world_size
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
        model: DDP,
        dataloader: StatefulDataLoader,
        trainer: BasicGPTTrainer,
        scheduler: LambdaLR,
        rank: int,
        world_size: int,
    ):
        # get trainer from

        self.setup(rank, world_size)

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
        for step, batch in tqdm(enumerate(dataloader)):
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

        self.cleanup()

    def save_checkpoint(
        self, step: int, trainer: BasicGPTTrainer, dataloader: StatefulDataLoader
    ):
        # only rank 0 should save the checkpoint
        if self.rank != 0:
            return

        checkpoint = trainer.get_state_dict(step=step)
        if checkpoint is None:
            return

        dataloader_state = dataloader.state_dict()
        dist.gather_object(
            obj=dataloader_state, object_gather_list=checkpoint["dataloader"], dst=0
        )
        checkpoint["dataloader"] = checkpoint["dataloader"][self.rank]

        # save the checkpoint to a file
        # 应该是不需要在配置里面写上路径的
        # 需要在在.cache/checkpoint_1000.pt ?
        # 那如果是多次训练呢？
        torch.save(checkpoint, f"checkpoint_{step}.pt")
