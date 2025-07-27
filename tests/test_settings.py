# 2025/7/20
# zhangzhong


from pathlib import Path

from xihe.settings import load_config


def test_example_conf() -> None:
    conf = load_config(conf_file=Path("example_conf.toml"))
    print(conf)
