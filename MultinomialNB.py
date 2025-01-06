import os
import torch
import pandas as pd
from tqdm import tqdm
from dataset import IMDb
from tfidf import TFIDF, visualize_feature_importance
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

if __name__ == "__main__":
    # 加载并预处理数据集
    train_data, test_data = IMDb(dataset_type='train'), IMDb(dataset_type='test')
    os.makedirs(os.path.join('..', 'outputs'), exist_ok=True)

    train_data.tokenize()
    train_data.build_vocab(min_freq=3)
    train_data.convert_tokens_to_ids(train_data.token2id)
    train_data.write_ids_pickle()

    test_data.tokenize()
    test_data.convert_tokens_to_ids(train_data.token2id)
    test_data.write_ids_pickle()

    train_token_ids = train_data.read_ids_pickle()
    test_token_ids = test_data.read_ids_pickle()
    print(pd.DataFrame(train_token_ids).shape)

    train_max_ids = 0
    train_X, train_Y = [], []
    for data in tqdm(train_token_ids, desc="Processing train dataset's token ids"):
        if max(data) > train_max_ids:
            train_max_ids = max(data)
        train_X.append(data)
    train_Y = train_data.data[1]

    test_X, test_Y = [], []
    for data in tqdm(test_token_ids, desc="Processing test dataset's token ids"):
        test_X.append(data)
    test_Y = test_data.data[1]

    # 特征提取
    tfidf = TFIDF(train_max_ids, token2id=train_data.token2id)
    tfidf.fit(train_X)
    train_F = tfidf.transform(train_X)
    test_F = tfidf.transform(test_X)
    print(train_F)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clf = MultinomialNB(alpha=0.001).fit(train_F, train_Y)
    
    # 在训练结束后调用可视化函数
    visualize_feature_importance(tfidf, train_F, train_Y)

    predicted_labels = clf.predict(test_F)

    # 计算分类器准确率
    print('准确率为：', metrics.accuracy_score(test_Y, predicted_labels))

