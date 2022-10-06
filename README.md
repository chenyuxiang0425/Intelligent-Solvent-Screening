# 简单的输入数据生成神经网络模型的demo
## 功能
输入训练集，自定义选择的模型类型，输出模型。根据预测集输出预测结果。

目前功能包括
- 特征向量选择
  - 随机森林
  - 极度随机树
- 数学模型
  - 支持向量机(SVM)
- 输出预测结果
  - 需要提供预测集

## 运行
```
> pip install -r requirements.txt
```

## 备注
- 可在 `models.py` 中自定义模型
- 可在 `main.py` 中调节 `train_model`、`test_model`、`output_data` 三个布尔变量来控制执行哪些任务
