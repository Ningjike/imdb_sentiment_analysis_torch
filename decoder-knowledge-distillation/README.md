# Decoder Knowledge Distillation
1. 通过对微调后的 Qwen-4B（Teacher）蒸馏迁移 Qwen3-0.6B（Student），对 IMDB 数据集进行情感分析二分类，达到了约为 94.31%的准确率。
2. 通过对微调后的 Deepseek（Teacher）蒸馏迁移 Qwen3-4B（Student），对 SST-2 数据集进行情感分析二分类，达到了约为 53.4%的准确率。
   
## 知识蒸馏原理
使用教师模型在相同输入下生成的 logits 作为“软标签”，并通过反向KL散度来衡量学生模型输出分布 logits 与教师模型输出分布 teacher_logits 之间的差异。
```
class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, if_use_entropy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        # 计算标准的监督损失
        loss = outputs_student.loss
        logits = outputs_student.logits
        
        with torch.no_grad():
            teacher_logits = teacher_outputs.logits
        # 如果教师模型和学生模型输出形状不匹配，对学生模型进行padding或对教师模型进行截断
        # print(logits.shape, teacher_logits.shape)
        # print(type(logits), type(teacher_logits))
        # if logits is None or teacher_logits is None:
        kl = 0
        if isinstance(logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
            # 对齐 logits 维度
            if logits.shape[-1] != teacher_logits.shape[-1]:
                # 截断 teacher_logits 与学生模型保持一致
                teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
                labels = inputs['labels']
                kl = compute_rkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0)
    
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        return (loss_total, outputs_student) if return_outputs else loss_total
```

## \<think> 蒸馏过程 7h 59m 19s · GPU T4 x2
指令格式：
```
alpaca_prompt = """You are an expert film critic. Analyze the following review step by step and explain why its sentiment is "{label}".

Review: {input}

<think>
Let's think step by step:
1. Look for emotionally charged words.
2. Assess the overall tone.
3. Justify why the sentiment is "{label}".
</think>"""
```
让teacher 模型、student 模型根据已知的训练数据的文本和标签进行推理，学习到推理能力

## 模型结果
<img width="700" height="99" alt="image" src="https://github.com/user-attachments/assets/685aecff-702a-4b20-ba1b-e7c7245f9f6b" />

模型效果不好的原因：
1. 可能是由于只训练了 1 个 epoch ,由于 kaggle 最大时长 12h 的限制, 只进行了 1 个 epoch 的 训练，模型还没学习到 sentence 和 label 的映射关系
2. 可能是由于 teacher 模型和 student 模型架构不同
3. 可能是由于指令格式设置不合理。

