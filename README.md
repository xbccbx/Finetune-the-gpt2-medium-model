本项目对gpt2-medium模型做了微调，使用美国总统Trump的推特记录作为数据集进行了训练。

- #### 运行环境

  - pytorch框架
  - 若进行训练，单卡需要约至少20G的显存。

- #### 如何运行

  - python train.py: 训练模型与Perplexity的计算。
  - python inference.py: 推理代码，由模型生成一条随机文本。
  - processed_dataset.csv为经过处理后的数据。

- #### 代码

  - 