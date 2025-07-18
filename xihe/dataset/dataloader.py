# 2025/7/17
# zhangzhong

# 我需要写一个可以按照一定的比例，混合多个数据集
# 带有shuffle
# 我们先用GPT2自带的tokenizer吧
# 之后再换成我们自己train出来的

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, IterableDataset


# 我们最终的目的是构造一个pytorch的Dataset
# 先不用想着DDP
# 先吧最简单的东西构造出来
# 那就先整一个数据集吧


tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.eos_token)

# 首先就只拿一个数据集进行测试吧
# 我们还是用stream模式
# 只不过是本地的stream
dataset = load_dataset(
    path="wikimedia/wikipedia",
    name="20231101.en",
    # split="train",
    split="train[:1000]",
)
assert isinstance(dataset, Dataset)

iterable_dataset: IterableDataset = dataset.to_iterable_dataset()


# 然后就是对数据进行tokenization
# huggingface里面的有tokenization的例子
# 咱们参考一下
# 我觉得可以写一个函数，然后做batch处理
# 然后处理完成的tokenization
# 然后在load的时候，在做packing
# 分开做，感觉更好一些
def tokenize(examples):
    return tokenizer(examples["text"])


iterable_dataset = iterable_dataset.map(
    function=tokenize,
    batched=True,
    remove_columns=iterable_dataset.column_names,
    drop_last_batch=True,
    # https://huggingface.co/docs/datasets/process#multiprocessing
    # Dataset是支持多进程处理map的
    # 但是stream不支持
    # num_proc=4,  # 可以根据CPU核心数调整
)

# TODO
# 不过我们可以手动做多进程支持
# 每个进程处理一个数据集，然后汇总到主进程，得到一个batch
# 再分发给不同的显卡
# 不过只有当我们的显卡占用率因为数据处理的太慢而占用率较低时，才应该使用

# 然后我们需要首先packing
# packing就是把一个batch里面的token 通过一个特殊的 eos token拼在一起
# 然后切分为1024的长度


def packing(examples):
    # 这里的examples是一个batch
    # 我们需要把每个example的tokens拼接起来
    # 然后切分为1024的长度
    packed_texts = []
    # for text in examples["input_ids"]:
    #     packed_text = tokenizer.eos_token.join()
    #     packed_texts.append(packed_text)
    packed_input_ids = tokenizer.eos_token.join

    # # 切分为1024的长度
    # packed_texts = [text[:1024] for text in packed_texts]

    # return {"input_ids": packed_texts}
    return examples


iterable_dataset = iterable_dataset.map(
    function=packing,
    batched=True,
    batch_size=8,  # 可以根据显存大小调整
    remove_columns=iterable_dataset.column_names,
    drop_last_batch=True,
)
