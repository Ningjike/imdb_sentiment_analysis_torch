# imdb_sentiment_analysis_torch
本项目基于kaggle竞赛“Bag of Words Meets Bags of Popcorn”,[(竞赛链接)](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview),进行nlp学习，实现电影评论文本的情感二分类任务。分别采用了cnn、transformer、gru等常见的神经网络模型，同时还尝试了Bert、distilBert、Roberta等大模型结合竞赛任务数据进行微调。同时利用deberta xxlarge模型进行情感分析，尝试了几种常见微调方式。
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
|Ptuning-int8|0.92268|
|Prompt|0.75600|

## fine tuning
1. LoRA： 5h 24m 29s · GPU T4 x2
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

2. Ptuning： 5h 37m 7s · GPU T4 x2
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
  
3. Prompt：6h 25m 9s · GPU P100
```
prompt_tuning_init_text = "Classify if the movie review is positive or negative.\n"
peft_config = PromptTuningConfig(
    num_virtual_tokens=10,
    task_type=TaskType.SEQ_CLS,
    prompt_tuning_init = PromptTuningInit.TEXT,
    prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path = model_id
)
```
问题： 
采用8bit量化出现RuntimeError: cublasLt ran into an error!故后来采用4bit量化，但模型训练3轮后准确率仅有0.57，几乎没有学到分类特征。

最终解决方案为：
- 限制max_length=128
- 限制per_device_train_batch_size=1
- 限制per_device_eval_batch_size=1
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
------
# 模型准确率
|models|准确率|
|---|---|
|DeBERTa-V2-unsloth|0.95844|
|ModernBERT-unsloth|0.95540|
|BERT-RDrop|0.93792|
|BERT-SCL-LoRA|0.92984|
|BERT-RDrop-LoRA|0.92516|
|BERT-SCL|0.91744|

## unsloth 使用
imdb_deberta_unsloth：5h 41m 47s · GPU T4 x2
imdb_modernbert_unsloth： 2h 57m 4s · GPU T4 x2
采用unsloth调用DeBERTa-V2-xxlarge模型：未采用8bit和4bit量化，且max_length设置为256，最终用时仍然在6h内完成了3轮训练，相比之前采用LoRA微调时不仅采用了量化，还进一步减少了max_length，仅完成了2轮训练，性能有所改进。

1. kaggle 安装 unsloth
```
!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
!pip install unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
```
2. 利用FastModel加载模型
```
model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    # 默认开启4bit量化
    load_in_4bit=False,
    max_seq_length=512,
    dtype=None,
    auto_model=AutoModelForSequenceClassification,
    num_labels=NUM_CLASSES,
    gpu_memory_utilization=0.5  # Reduce if out of memory
)
```
3. LoRA微调配置
```
model = FastModel.get_peft_model(
    model,
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=32,  # Recommended alpha == r at least
    lora_dropout=0.05,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    use_gradient_checkpointing="unsloth",  # Reduces memory usage
    target_modules = "all-linear", # Optional now! Can specify a list if needed
    task_type="SEQ_CLS",
)
```
## R-Drop
参考[论文](https://arxiv.org/abs/2106.14448)

<img width="944" height="438" alt="image" src="https://github.com/user-attachments/assets/24da268f-0d33-4f3d-a665-6aa01a2edf93" />

R-Drop能够降低基于dropout模型在训练与推理阶段之间的一致性差异。具体来讲，对于每个输入均进行两次前向传播，分别采用两个随机Dropout子模型进行预测，采用KL散度来衡量差异程度，同时结合交叉熵损失构建总损失函数。
<img width="1175" height="379" alt="image" src="https://github.com/user-attachments/assets/4aaf1c5f-4d56-4467-9914-18600db1d868" />

继承Huggingface的BertPreTrainedModel重写forward方法：
```
class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        total_loss = None
        
        if labels is not None:
            kl_outputs = self.bert(input_ids, attention_mask, token_type_ids)
            kl_output = kl_outputs[1]
            kl_output = self.dropout(kl_output)
            kl_logits = self.classifier(kl_output)
            # 计算第一次前向的交叉熵损失Lnll1
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 计算第二次前向的交叉熵损失Lnll2
            ce_loss = loss_fct(kl_logits.view(-1, self.num_labels), labels.view(-1))
            # 计算Lkl
            kl_loss = (KL(logits, kl_logits, "sum") + KL(kl_logits, logits, "sum")) / 2.
            # 总损失=Lnll + Lkl
            total_loss = loss + ce_loss + kl_loss

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
```
## Supervised Constrative Learning
参考[论文](https://arxiv.org/abs/2004.11362)
<img width="717" height="668" alt="image" src="https://github.com/user-attachments/assets/c3a02223-028f-4816-9474-bc89e9f7037d" />

<img width="1037" height="128" alt="image" src="https://github.com/user-attachments/assets/828de657-3cc1-4fee-9851-14c73fda524d" />



