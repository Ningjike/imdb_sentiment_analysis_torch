# Decoder Knowledge Distillation
通过对微调后的 Qwen-4B（Teacher）蒸馏迁移 Qwen3-0.6B（Student），对 IMDB 数据集进行情感分析二分类，达到了约为 94.31%的准确率。

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

## deepseek \<think>过程蒸馏
利用deepseek进行指令学习，得到训练数据的推理过程记录，但由于训练数据过多（25000条），该过程运行时长远超 7h，最终未能实现
```
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "/kaggle/input/corpus-imdb/labeledTrainData.tsv"
OUTPUT_PATH = "imdb_deepseek_cot_fast.csv"

BATCH_SIZE = 4
MAX_NEW_TOKENS = 256
NUM_SAMPLES = None


# 构造 prompt
def build_prompt(review: str, true_label: str) -> str:
    return f"""You are an expert film critic. Given the following movie review and its correct sentiment label, 
                generate a clear, step-by-step reasoning that explains why the sentiment is "{true_label}".
                
                Review: {review}
                
                Correct Sentiment: {true_label}
                
                <think>
                Let's analyze step by step:
                1. Identify key emotional words or phrases.
                2. Consider tone, context, and possible sarcasm.
                3. Explain how these lead to the conclusion of "{true_label}".
                </think>
                
                Now, provide your reasoning inside  <think>  and  </think>  tags only.
                """

# 提取 reasoning
def extract_reasoning(text: str) -> str:
    # 移除开头可能存在的 </think>
    if text.strip().startswith("</think>"):
        text = text[len("</think>") :].lstrip()
    
    start_tag = "<think>"
    end_tag = "</think>"
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag, start_idx + len(start_tag) if start_idx != -1 else 0)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx + len(start_tag) : end_idx].strip()
    else:
        return text.strip()

def main():
    df = pd.read_csv(DATA_PATH, delimiter="\t", quoting=3)
    if NUM_SAMPLES is not None:
        df = df.head(NUM_SAMPLES)

    # 构建 prompts
    prompts = []
    sentiments = []
    for _, row in df.iterrows():
        label = "positive" if row["sentiment"] == 1 else "negative"
        prompts.append(build_prompt(row["review"], label))
        sentiments.append(row["sentiment"])

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    results = []

    # 批量生成
    raw_outputs = generator(
        prompts,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    flat_outputs = [out[0] if isinstance(out, list) else out for out in raw_outputs]

    # 后处理
    for i, out in enumerate(flat_outputs):
        reasoning = extract_reasoning(out["generated_text"])
        results.append({
            "review": df.iloc[i]["review"],
            "reasoning": reasoning,
            "sentiment": sentiments[i]
        })

    # 保存结果
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
```

