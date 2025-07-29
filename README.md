# XiHe

My LLM.

## how to use

1. write your wandb api key to `.env` file, like this:

   ```bash
   WANDB_API_KEY=your_wandb_api_key
   ```

2. uv run pre-commit install

## TODO

1. 2025/7/19
   1. [x] 参考两份代码示例，确保实现是正确且高效的，并且编写单元测试
   2. [x] 实现dataloader，正确的采样多种数据集
   3. [x] 实现Trainer
2. 2025/7/20
   1. [x] 可以用小批量数据在小规模模型上跑通
3. 2025/7/24
   1. [x] 添加cosine scheduler的单元测试，应该用ipynb展示出来更合适
   2. [x] 实现混合精度训练， amp
   3. [x] 适配wandb
   4. [x] 实现DDP
4. 2025/7/25
   1. [x] 实现模型保存和加载 <https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html>，需要确认所有需要保存的状态和如何加载这些状态。以及一个保存模型的策略
   2. [x] 自顶向下重构代码框架
5. 2025/7/26
   1. [x] 自底向上的添加单元测试并重构
6. 2025/7/27
   1. [x] 确认dataloader的实现是正确的，需要给出几个简单的dataset的例子，然后走我们自己的数据处理管线，看看是否正确
   2. [x] 加上ruff等各种静态检查工具, pre-commit, mypy等
   3. [x] 确认Trainer的实现是正确的，包括amp，ddp等
   4. [x] 实现各种metrics的统计，比如处理token的速率等等
   5. [x] redesign chekcpoint
7. 2025/7/28
   1. [x] add unit test for checkpoint
   2. [x] 最终确认所有测试用例通过
   3. [x] 确定XiHe 120M 模型的各种参数
   4. [x] 添加模型训练速度的测试
8. 2025/7/29
   1. [ ] 优化模型实现
   2. [ ] 实现梯度累积 gradient accumulation steps
   3. [ ] 添加数据处理速度的测试
   4. [ ] 开始模型分布式训练！<https://docs.pytorch.org/tutorials/intermediate/ddp_series_minGPT.html> 又找到一个代码参考源，不过这个也是咱们写完之后对答案用吧
