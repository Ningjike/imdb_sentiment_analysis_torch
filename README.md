# imdb_sentiment_analysis_torch
本项目基于kaggle竞赛“Bag of Words Meets Bags of Popcorn”,[(竞赛链接)](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview),进行nlp学习，实现电影评论文本的情感二分类任务。分别采用了cnn、transformer、gru等常见的神经网络模型，同时还尝试了Bert、distilBert、Roberta等大模型结合竞赛任务数据进行微调。
# 神经网络模型
## 各模型评估结果
|模型|准确率|
|---|---|
|Transformer|0.55360|
|CNN|0.72060|
|CNN-LSTM|0.77996|
|Attention-LSTM|0.80700|
|Capsule-LSTM|0.84324|
|GRU|0.85724|
|LSTM|0.87824|
使用 Gensim 加载 GloVe，需要将 GloVe 格式的词向量文件转换为 Gensim 兼容的 Word2Vec 文本格式，见glove-gensim.py

# 预训练模型
|预训练模型|准确率|
|---|---|
|BERT-native|0.90112|
|DistilBERT-native|0.90772|
|DistilBERT-trainer|0.92860|
|BERT-scratch|0.93516|
|BERT-trainer|0.93848|
|RoBERTa-trainer|0.95232|
kaggle有时网络连接不稳定，加载模型可能会出现HTTPStatusError，故部分代码从本地导入模型

# DeBERTa xxlarge PEFT
## 模型准确率
|模型|准确率|
|---|---|
|LoRA-int8|0.93752|
|Prompt|0.56360|
|Ptuning-int8|0.92268|
## fine tuning
1. LoRA
```
lora_config = LoraConfig(
    # 低秩矩阵的秩
    r=16,
    # 缩放因子
    lora_alpha=32,
    target_modules=['query_proj', 'value_proj'],
    lora_dropout=0.05,
    bias="none",
    # 任务类型为文本分类任务
    task_type=TaskType.SEQ_CLS
)
```
该部分出现OutOfMemoryError、且训练时间需要10几个小时超出kaggle最长支持运行时间9h
解决方案：
- 采用8bit量化
- 限制max_length=128
- 训练num_train_epochs=2
2. Prompt
```

```
3. Ptuning
```
# Define PromptEncoder Config
peft_config = PromptEncoderConfig(
    num_virtual_tokens=20,
    encoder_hidden_size=128,
    task_type=TaskType.SEQ_CLS
)
```
该部分出现OutOfMemoryError、且训练时间需要10几个小时超出kaggle最长支持运行时间9h
解决方案：
- 采用8bit量化
- 限制max_length=128
- 训练num_train_epochs=2

4. Prefix
```
# Define Prefix Config
peft_config = PrefixTuningConfig(
    num_virtual_tokens=20,
    task_type=TaskType.SEQ_CLS
)
```
问题：
- ValueError: Prefix tuning does not work with gradient checkpointing.

  尝试手动关闭gradient checkpointing
  ```
  model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
  ```
  
- ValueError: Model does not support past key values which are required for prefix tuning.

  说明模型不支持​​ past key values​​，而这是前缀调整的必要条件，因此Prefix Tuning 无法用于 DeBERTa-v2。

add adaptor
```
model = get_peft_model(model, xxxx_config)
```
