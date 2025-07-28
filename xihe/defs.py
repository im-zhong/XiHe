# 2025/7/25
# zhangzhong

from pathlib import Path


class Defs:
    @property
    def cache_dir(self) -> Path:
        return Path(".cache")

    # @property
    # def ckpt_dir(self) -> Path:
    #     return self.cache_dir / "ckpts"

    # def get_ckpt_path(self, project: str, step: int) -> Path:
    #     project_dir = self.ckpt_dir / project
    #     project_dir.mkdir(parents=True, exist_ok=True)
    #     return project_dir / f"ckpt_{step}.tar"

    # # 所有构造checkpoint路径相关的部分都放在一起
    # # 这里感觉只保留cache的路径构造就够了
    # # checkpoint path的路径的构造放到checkpoint模块里面

    # # 毕竟也可能没有
    # def get_best_ckpt_path(self, project: str) -> Path | None:
    #     project_dir = self.ckpt_dir / project
    #     project_dir.mkdir(parents=True, exist_ok=True)
    #     # 我想从名字里直接看到最优模型的step是多少
    #     return project_dir / "best_best.tar"


defs = Defs()
