# XiHe

My LLM.

## TODO

1. 2025/7/19
   1. [x] 参考两份代码示例，确保实现是正确且高效的，并且编写单元测试
   2. [x] 实现dataloader，正确的采样多种数据集
   3. [x] 实现Trainer
2. 2025/7/20
   1. [x] 可以用小批量数据在小规模模型上跑通
3. 2025/7/24
   1. [x] 添加cosine scheduler的单元测试，应该用ipynb展示出来更合适
   2. [ ] 添加dataloader的单元测试，可能需要设置drop last batch
   3. [ ] 实现混合精度训练， amp？
   4. [ ] 适配wandb
   5. 实现DDP
   6. 实现模型保存和加载 <https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html>
   7. 确定XiHe 120M 模型的各种参数
   8. 开始模型分布式训练！<https://docs.pytorch.org/tutorials/intermediate/ddp_series_minGPT.html> 又找到一个代码参考源，不过这个也是咱们写完之后对答案用吧
