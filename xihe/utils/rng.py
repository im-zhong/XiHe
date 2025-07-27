# # 2025/7/25
# # zhangzhong

# import random

# import numpy as np
# import torch


# # 感觉这一部分不是特别的有必要。。。
# # 增加了代码的复杂度
# def set_all_seeds(seed: int):
#     # Python 内置随机数
#     random.seed(seed)

#     # NumPy 随机数
#     np.random.seed(seed)

#     # PyTorch CPU
#     torch.manual_seed(seed)

#     # PyTorch 所有 GPU
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#     # 让 cudnn 的算法结果 deterministic（可选）
#     # torch.backends.cudnn.deterministic = True
#     # torch.backends.cudnn.benchmark = False


# def set_all_rng_states(rng_states):
#     random.setstate(rng_states["python"])
#     np.random.set_state(rng_states["numpy"])
#     torch.set_rng_state(rng_states["torch_cpu"])
#     if rng_states["torch_cuda"] is not None:
#         torch.cuda.set_rng_state_all(rng_states["torch_cuda"])


# def get_all_rng_states():
#     rng_states = {
#         "python": random.getstate(),
#         "numpy": np.random.get_state(),
#         "torch_cpu": torch.get_rng_state(),
#         "torch_cuda": torch.cuda.get_rng_state_all()
#         if torch.cuda.is_available()
#         else None,
#     }
#     return rng_states
