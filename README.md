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
尝试使用unsloth改进微调效率，对R-Drop、Supervised Contrastive Learning方法进行实践。
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
   若出现ImportError: Unsloth: Please install unsloth_zoo via `pip install unsloth_zoo`可尝试
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

<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/24da268f-0d33-4f3d-a665-6aa01a2edf93" />

R-Drop能够降低基于dropout模型在训练与推理阶段之间的一致性差异。具体来讲，对于每个输入均进行两次前向传播，分别采用两个随机Dropout子模型进行预测，采用KL散度来衡量差异程度，同时结合交叉熵损失构建总损失函数。
<img width="800" height="250" alt="image" src="https://github.com/user-attachments/assets/4aaf1c5f-4d56-4467-9914-18600db1d868" />

### 继承Huggingface的BertPreTrainedModel重写forward方法：
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

- 监督对比特点：
    每个锚点：多个正样本对、多个负样本对
    正样本来自于与锚点相同类别的样本
  
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/c3a02223-028f-4816-9474-bc89e9f7037d" />

<img width="800" height="90" alt="image" src="https://github.com/user-attachments/assets/828de657-3cc1-4fee-9851-14c73fda524d" />

### 损失函数设计
```
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # 此时为有监督学习
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # 构建正样本掩码，属于同一类的为正样本
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # 锚点的选择
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # 计算锚点样本与负样本的余弦相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss，计算最终loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
```
### 继承Huggingface的BertPreTrainedModel重写forward方法：
```
class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = 0.2

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

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            scl_fct = SupConLoss()
            scl_loss = scl_fct(pooled_output, labels)

            loss = ce_loss + self.alpha * scl_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

```
# LoRA微调
原尝试使用unsloth调用重写后的模型，结果出现多次形如下方的报错：

AttributeError: 'BertWithRDrop' object has no attribute 'xxxxx'

自定义模型不太兼容unsloth，尝试修改自定义模型，最终未能成功。
考虑到进行微调的BERT模型相比DeBERTa-V2-xxlarge，规模相对较小，训练时长可以接受，故直接采用PEFT进行LoRA微调。
- imdb_bert_drop_lora: 1h 57m 14s · GPU T4 x2
```
config = AutoConfig.from_pretrained(model_name)
config.num_labels = NUM_CLASSES

model = BertWithRDrop(config)

pretrained_bert = AutoModel.from_pretrained(model_name)
model.bert = pretrained_bert

tokenizer = AutoTokenizer.from_pretrained(model_name)


peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"],  # BERT 的注意力层
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

- imdb_bert_scl_lora: 1h 26m 37s · GPU T4 x2
```
config = AutoConfig.from_pretrained(model_name)
config.num_labels = NUM_CLASSES

model = BertWithSCL(config, alpha=0.2)

pretrained_bert = AutoModel.from_pretrained(model_name)
model.bert = pretrained_bert

tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"],  # BERT 的注意力层
)

model = get_peft_model(model, peft_config)
```
