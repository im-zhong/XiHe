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
   1. [x] 优化模型实现，包括RMSNOrm，RoPE，以及我们其他模块的实现，改变一下风格，看看是否对显存和速度有影响
   2. [x] 实现梯度累积 gradient accumulation steps
   3. [x] 开始模型分布式训练！<https://docs.pytorch.org/tutorials/intermediate/ddp_series_minGPT.html> 又找到一个代码参考源，不过这个也是咱们写完之后对答案用吧
9. 2025/7/31
   1. [ ] 添加数据处理速度的测试
   2. [x] 统一setting里面的batch size
   3. [x] 数据处理阶段的map的batch size最好也可以设置
   4. [x] 添加日志模块，每个进程独立输出到自己的日志文件，我好方便调试，判断程序是不是正确运行。在重构的过程中逐渐的添加日志吧。
   5. [x] 现在为了方便，trainer no_sync部分的代码通不过pyright，要想个办法改正过来。把这个该镇过来，所有的测试都应该通过。再把第一个任务做了。都做完了，去锻炼。回来重构。
   6. [x] 现在每个进程都会登陆wandb，应该改成只有rank0需要登陆
   7. [x] 启动光是加载数据就加载好久啊。。。不行啊，联网读取数据实在是不靠谱，还是在本地启用streaming模式是最好的。
   8. [x] create dataset根据streaming模式判断怎么获取features name
   9. [x] 查清并解决从本地加载数据集非常慢的问题，而且每个进程都会加载一次也太奇怪了，chatgpt说可以先加载然后dataset.save_to_disk，然后load_from_disk咱们可以试一下。而且好像显示指定了cache_dir加载也快了一些了。第一次加载之后，之后再加载会稍快一些，但是还是很慢。
   10. [x] 为什么wandb会登陆四次呢？应该只在rank=0上处理任何与wandb有关的逻辑才对
   11. [x] dataset处理数据的batch size太小了，导致每个step都需要读取数据进行处理，咱们把batch size搞大一点
   12. [ ] 搞清楚为什么gpu0占用的显存会多一些。
   13. [ ] Token indices sequence length is longer than the specified maximum sequence length for this model (8668 > 1024). Running this sequence through the model will result in indexing errors 为什么会有这个提示？
   14. [ ] PackingDataset有冗余，重构一下
   15. [ ] 上面的都改完之后，每个模块都重构一下吧。
