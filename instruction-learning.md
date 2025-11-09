# 指令学习
|模型|准确率|
|---|---|
|deepseek| 0.50108|
|gemma|0.50736|
|Phi-4-mini|0.52672|
|Qwen3|0.54636|
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B：4h 26m 50s · GPU T4 x2
  2B params
- google/gemma-2-2b-it：6h 41m 38s · GPU T4 x2
  3B params
- microsoft/Phi-4-mini-instruct：10h 10m 17s · GPU T4 x2
  4B params
- Qwen/Qwen3-1.7B：4h 27m 28s · GPU T4 x2
  3B params
## prompt
```
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
### Instruction: 
You are an expert of film comment analysis. Analyze the given text from an online review and determine the sentiment polarity. Return a single number of either 0 and 1, with 0 being negative and 1 being the positive sentiment.  
### Input: {}
### Response:
<think>
"""
```
生成回答形式：
'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

\### Instruction: 
You are an expert of film comment analysis. Analyze the given text from an online review and determine the sentiment polarity. Return a single number of either 0 and 1, with 0 being negative and 1 being the positive sentiment.  

\### Input: "This movie is a disaster within a disaster film. It is full of great action scenes, which are only meaningful if you throw away all sense of reality. Let\'s see, word to the wise, lava burns you; steam burns you. You can\'t stand next to lava. Diverting a minor lava flow is difficult, let alone a significant one. Scares me to think that some might actually believe what they saw in this movie.Even worse is the significant amount of talent that went into making this film. I mean the acting is actually very good. The effects are above average. Hard to believe somebody read the scripts for this and allowed all this talent to be wasted. I guess my suggestion would be that if this movie is about to start on TV ... look away! It is like a train wreck: it is so awful that once you know what is coming, you just have to watch. Look away and spend your time on more meaningful content."

\### Response:

<think>
  
</think>


0'

因此，提取回答方法：
```
for i, response in enumerate(tqdm(responses, desc="analysis", total=total_samples)):
    output = response[0]["generated_text"]
    # 提取数字
    digits = re.findall(r'[01]', output)
    pred = int(digits[-1]) if digits else 0
    test_pred.append(pred)
```
