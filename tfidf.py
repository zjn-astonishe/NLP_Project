import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import seaborn as sns

class TFIDF:
    def __init__(self, vocab_size, token2id=None, norm='l2', smooth_idf=True, sublinear_tf=True):
        self.vocab_size = vocab_size
        self.norm = norm
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.token2id = token2id if token2id is not None else {}
        self.id2token = {v: k for k, v in self.token2id.items()}  # 假设token2id是{k: v}形式的字典
    
    def fit(self, X):
        doc_freq = np.zeros(self.vocab_size, dtype=np.float64)
        print()
        for data in tqdm(X, desc="Processing tf-idf fit"):
            for token_id in set(data):
                doc_freq[token_id-1] += 1
        doc_freq += int(self.smooth_idf)
        n_samples = len(X) + int(self.smooth_idf)
        self.idf = np.log(n_samples / doc_freq) + 1

    def transform(self, X):
        assert hasattr(self, 'idf')
        term_freq = np.zeros((len(X), self.vocab_size), dtype=np.float64)
        for i, data in enumerate(X):
            for token in data:
                term_freq[i, token - 1] +=1
        if self.sublinear_tf:
            term_freq = np.log(term_freq+1)
        Y = term_freq * self.idf
        if self.norm:
            row_norm = (Y**2).sum(axis=1)
            row_norm[row_norm==0] = 1
            Y /= np.sqrt(row_norm)[:, None]
        return Y
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def get_feature_names(self):
        """Return a list of feature names corresponding to the vocabulary."""
        return [self.id2token.get(i+1, f"UNK_{i}") for i in range(self.vocab_size)]
    
def visualize_feature_importance(tfidf, train_X, train_Y):
    # 计算特征重要性
    features = tfidf.get_feature_names()
    chi2_scores, p_values = chi2(train_X, train_Y)
    
    # 将得分和对应的特征名组合成元组列表
    feature_importances = list(zip(features, chi2_scores))
    
    # 按照得分降序排序
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    
    # 取前N个特征进行可视化
    N = 10  # 可以根据需要调整这个数字
    top_features = feature_importances[:N]
    words, scores = zip(*top_features)

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(scores), y=list(words))
    plt.title('Top Feature Importances')
    plt.xlabel('Chi-Square Score')
    plt.ylabel('Words')
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')  # 保存为PNG格式
    plt.close()

    print("Feature importance visualization saved.")