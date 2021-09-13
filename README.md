先将AutoMaster_TrainSet 和 AutoMaster_TestSet 拷贝到data 路径下 再使用 .



代码结构

+ result 结果保存路径
    ....    
+ seq2seq_tf2 模型结构
    ....
+ utils 工具包
    + config  配置文件
    + data_loader 数据处理模块
    + multi_proc_utils 多进程数据处理
+ data  数据集
    + AutoMaster_TrainSet 拷贝数据集到该路径
    + AutoMaster_TestSet  拷贝数据集到该路径
    ....
 
 训练步骤:
 1. 拷贝数据集到data路径下
 2. 运行utils\data_loader.py可以一键完成 预处理数据 构建数据集
 
 ##  seq2seq_tf2 模块
 * 训练模型 运行seq2seq_torch\train.py脚本,进入 summary 目录,运行如下命令:
     ```bash
     $ python -m src.seq2seq_torch.train
     ```   
 预测步骤:
 1. greedy decode 和 beam search 的代码都在 predict_helper.py 中，greedy结果
'python -m src.seq2seq_torch.predict greedy_decode=True'
```json    
{
  "rouge-1": {
    "r": 0.20962963528826314,
    "p": 0.48632813895381194,
    "f": 0.2736560922579011
  },
  "rouge-2": {
    "r": 0.05910192048596341,
    "p": 0.09515444736172018,
    "f": 0.06712834448906212
  },
  "rouge-l": {
    "r": 0.19700879589229262,
    "p": 0.4530833814442011,
    "f": 0.2562386261778882
  }
}
```
2、运行 predict.py 调用 greedy decode 或者 beam search 进行预测
'python -m src.seq2seq_torch.predict greedy_decode=False'
```json 
{
  "rouge-1": {
    "r": 0.10134758008519588,
    "p": 0.38007240081166993,
    "f": 0.1478587141500604
  },
  "rouge-2": {
    "r": 0.008943314071105575,
    "p": 0.04371892992562038,
    "f": 0.013845235652478319
  },
  "rouge-l": {
    "r": 0.09593903477975994,
    "p": 0.3594068694966316,
    "f": 0.1397642882930187
  }
}

```