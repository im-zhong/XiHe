# 2025/7/26
# zhangzhong

import time

from tqdm import tqdm

from xihe.dataset import create_dataloader
from xihe.schemas import DatasetArgs
from xihe.tokenizer import create_tokenizer


def test_create_dataloader() -> None:
    tokenizer = create_tokenizer("gpt2")
    datasets_args = [
        DatasetArgs(
            path="allenai/c4",
            name="en",
            # streaming=True, 下，不支持slice
            # split="train[:1024]",
            split="train",
            num_epochs=1,
            streaming=True,
        ),
        DatasetArgs(
            path="wikimedia/wikipedia",
            name="20231101.en",
            # split="train[:1024]",
            split="train",
            num_epochs=2,
            streaming=True,
        ),
    ]

    dataloader = create_dataloader(
        tokenizer=tokenizer,
        rank=0,
        batch_size=4,
        map_batch_size=1024,
        context_length=512,
        world_size=4,
        seed=42,
        datasets_args=datasets_args,
        sampling_probabilities=[0.5, 0.5],  # Example sampling probabilities
    )

    assert dataloader is not None
    print("Dataloader created successfully.")


def dataset_generate_tokens_speed(streaming: bool, testing_steps: int) -> None:  # noqa: FBT001
    # 我好像发现了一种神奇的使用方法
    # 就是设置HF_HUB_OFFLINE=1同时streaming=True
    # hf就会直接从本地加载数据，但是不进行处理，就会加载的非常快。大概是这样？真的是这样！
    tokenizer = create_tokenizer("gpt2")
    datasets_args = [
        # DatasetArgs(
        #     path="allenai/c4",
        #     name="en",
        #     # streaming=True, 下，不支持slice
        #     # split="train[:1024]",
        #     split="train",
        #     num_epochs=1,
        #     streaming=streaming,
        # ),
        # DatasetArgs(
        #     path="wikimedia/wikipedia",
        #     name="20231101.en",
        #     # split="train[:1024]",
        #     split="train",
        #     num_epochs=2,
        #     streaming=streaming,
        # ),
        # 咱们换成两个小的数据集，最起码local的测试可以跑起来
        DatasetArgs(
            path="nampdn-ai/tiny-codes",
            name=None,
            split="train",
            num_epochs=2,
            streaming=streaming,
        ),
        DatasetArgs(
            path="common-pile/arxiv_papers",
            name=None,
            split="train",
            num_epochs=1,
            streaming=streaming,
        ),
    ]

    dataloader = create_dataloader(
        tokenizer=tokenizer,
        rank=0,
        batch_size=128,
        map_batch_size=8192,
        context_length=1024,
        world_size=4,
        seed=42,
        datasets_args=datasets_args,
        # sampling_probabilities=[0.5, 0.5],  # Example sampling probabilities
    )

    # 统计时间
    start_time = time.time()
    total_tokens = 0
    for i, batch in tqdm(enumerate(dataloader), total=testing_steps):
        tokens = batch["input_ids"]
        total_tokens += tokens.shape[0] * tokens.shape[1]
        if i >= testing_steps - 1:
            break
    end_time = time.time()
    print(
        f"Streaming dataset generated {total_tokens} tokens in {end_time - start_time} seconds. cosuming speed: {total_tokens / (end_time - start_time)} tokens/second"
    )


# TODO: 还可以测试一个东西啊
# 就是我们使用流式处理数据的速度
# 还有模型能够吃的token的速度
# 还有本地读取文件处理数据的速度
# 这些都应该测试一下
# 那咱们就可以把所有数据集的定义放到global变量里面，因为也有很多地方都在使用了
# 这两个函数应该用同一份实现，然后用
def test_streaming_dataset_generate_token_speed() -> None:
    # 13058.17316471823 tokens/second
    dataset_generate_tokens_speed(streaming=True, testing_steps=10)


def test_local_dataset_generate_tokens_speed() -> None:
    # 14535.950802919884 tokens/second
    dataset_generate_tokens_speed(streaming=False, testing_steps=10)
