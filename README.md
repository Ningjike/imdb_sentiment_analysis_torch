# imdb_sentiment_analysis_torch
本项目基于kaggle竞赛“Bag of Words Meets Bags of Popcorn”,[(竞赛链接)](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview),进行nlp学习，实现电影评论文本的情感二分类任务。分别采用了cnn、transformer、gru等常见的神经网络模型，同时还尝试了Bert、distilBert、Roberta等大模型结合竞赛任务数据进行微调。
# 各模型评估结果
|模型|准确率|
|---|---|
|LSTM|0.50000|
|Transformer|0.56360|
|CNN|0.75880|
|Attention-LSTM|0.80700|
|CNN-LSTM|0.83476|
|GRU|0.83844|
|Capsule-LSTM|0.88408|
|BERT-native|0.90112|
|DistilBERT-native|0.90772|
|DistilBERT-trainer|0.92860|
|BERT-scratch|0.93516|
|BERT-trainer|0.93848|
|RoBERTa-trainer|0.95232|
