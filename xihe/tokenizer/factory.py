# 2025/7/26
# zhangzhong

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer


def create_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)
