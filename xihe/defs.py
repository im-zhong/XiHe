# 2025/7/25
# zhangzhong

from pathlib import Path
import os


class Defs:
    @property
    def cache_dir(self) -> Path:
        return Path(".cache")

    @property
    def ckpt_dir(self) -> Path:
        return self.cache_dir / "ckpts"

    def get_ckpt_path(self, project: str, step: int) -> Path:
        project_dir = self.ckpt_dir / project
        os.makedirs(name=project_dir, exist_ok=True)
        return project_dir / f"ckpt_{step}.tar"


defs = Defs()
