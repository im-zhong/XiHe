# XiHe

My LLM.

## how to use

1. write your wandb api key to `.env` file, like this:

   ```bash
   WANDB_API_KEY=your_wandb_api_key
   ```

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
   2. [ ] 添加单元测试并重构
   3. 确定XiHe 120M 模型的各种参数
   4. 开始模型分布式训练！<https://docs.pytorch.org/tutorials/intermediate/ddp_series_minGPT.html> 又找到一个代码参考源，不过这个也是咱们写完之后对答案用吧
