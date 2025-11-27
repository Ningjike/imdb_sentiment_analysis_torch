# 知识蒸馏
## 一、 模型效果
| model | accuracy |
|--------|--------|
| teacher: deberta / student: part of teacher | 0.88384 |
| teacher: deberta / student: tiny_bert | 0.50000 |
| teacher: bert / student: tiny_bert | 0.86492 |

## 二 、模型构建
### 初始化学生模型：
1. 提取教师模型的部分层：
```
def create_student_from_teacher(teacher_model):
    # 创建学生配置
    teacher_config = teacher_model.config
    student_config = DebertaV2Config.from_dict(teacher_config.to_dict())
    student_config.num_hidden_layers = teacher_config.num_hidden_layers // 2

    # 创建学生模型
    student_model = type(teacher_model)(student_config)

    # 复制 embedding 层
    student_model.deberta.embeddings.load_state_dict(teacher_model.deberta.embeddings.state_dict())

    # 复制 pooler
    if (
        hasattr(teacher_model.deberta, "pooler")
        and teacher_model.deberta.pooler is not None
        and student_model.deberta.pooler is not None
    ):
        student_model.deberta.pooler.load_state_dict(teacher_model.deberta.pooler.state_dict())

    # 复制 encoder
    num_student_layers = student_config.num_hidden_layers
    for i in range(num_student_layers):
        student_model.deberta.encoder.layer[i].load_state_dict(teacher_model.deberta.encoder.layer[i].state_dict())

    # 复制分类头
    student_model.classifier.load_state_dict(teacher_model.classifier.state_dict())

    return student_model
```
2. 直接使用 tiny-bert 作为学生模型：
```
student_model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
```
### 构建蒸馏训练框架
```
class DistillationTrainingArguments(TrainingArguments): 
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.alpha = alpha 
        self.temperature = temperature 
        
class DistillationTrainer(Trainer): 
    def __init__(self, *args, teacher_model=None, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.teacher = teacher_model 
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device) 
        self.teacher.eval() 
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None) : 
        labels = inputs.pop("labels")
        # compute student output
        outputs_student = model(**inputs)

        # compute teacher output
        with torch.no_grad(): 
            outputs_teacher = self.teacher(**inputs) 
            
        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()
        
        # 计算硬标签损失
        student_loss = F.cross_entropy(outputs_student.logits, labels)
        
        # Soften probabilities and compute distillation loss
        # 计算 KL 散度
        loss_function = nn.KLDivLoss(reduction="batchmean") 
        
        # 计算学生模型和教师网络数值输出的 KL 散度
        loss_logits = (loss_function(
            # logits首先除以 temperature, 以增加对错误分类的关注
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1), 
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2)) 
        
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits 
        return (loss, outputs_student) if return_outputs else loss

```
DistillationTrainingArguments、DistillationTrainer 分别继承自 transformer 的 TrainingArguments 及 Trainer ，引入了蒸馏训练中的两个重要参数： temperature 及 alpha ，同时重写了损失计算逻辑。

#### 蒸馏训练的损失计算流程
1. 学生模型前向传播：outputs_student.logits;
2. 教师模型前向传播：outputs_teacher.logits;
3. 使用 outputs_student.logits 与真实标签 labels 计算硬标签损失
4. 计算 temperature 缩放后的 outputs_student.logits 与 outputs_teacher.logits 的 KL 散度作为蒸馏损失
5. 求加权和
   
注： 由于自定义的 compute_loss 中对模型输出（logits）进行了多次使用，导致 PyTorch 在 checkpoint 机制下无法正确保留中间激活值，在 arguments 中 gradient_checkpointing 若设置为 True, 可能出现 “RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.”类似错误。

同时在调用学生模型时：
```
if hasattr(student_model, "gradient_checkpointing_disable"):
    student_model.gradient_checkpointing_disable()
else:
    student_model.config.gradient_checkpointing = False
``` 
   
