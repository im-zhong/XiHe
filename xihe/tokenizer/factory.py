# 2025/7/26
# zhangzhong

from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


def create_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)
