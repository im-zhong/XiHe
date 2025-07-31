# 2025/7/30
# zhangzhong

from datasets import load_dataset, load_from_disk

# 加载wiki吧 稍微小一点
dataset = load_dataset(
    path="wikimedia/wikipedia",
    name="20231101.en",
    split="train",
    streaming=False,
    cache_dir="/data2/huggingface/datasets",
    num_proc=8,
)

dataset.save_to_disk("wiki_dataset")  # type: ignore
# 就算直接加载这个也不是很快
# 看来测试的时候，少加载几个数据集吧，实际训练的时候再全部加载吧
# 每次停止训练，重新启动就需要耗费一个小时
# 以后再做预训练，数据加载这块自己写吧
dataset = load_from_disk("wiki_dataset")
