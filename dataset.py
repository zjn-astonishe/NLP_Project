import os
import nltk
import re
import pickle
import torch
import json
import numpy as np
from tqdm import tqdm
# import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from download import download_imdb
from split_aclimdb_dataset import split_aclimdb_dataset

# 使用PyTorch的DataLoader来进行数据循环，因此按照PyTorch的接口
# 实现myDataset和DataCollator两个类
# myDataset是对特征向量和标签的简单封装便于对齐接口，
# DataCollator用于批量将数据转化为PyTorch支持的张量类型
class myDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

class DataCollator:
    @classmethod
    def lr_collate_batch(cls, batch):
        feats, labels = [], []
        for x, y in batch:
            feats.append(x)
            labels.append(y)
        # 直接将一个ndarray的列表转化为张量是非常慢的，
        # 所以需要提前将列表转化为一整个ndarray
        feats = torch.tensor(np.array(feats), dtype=torch.float)
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        return {'input_feats': feats, 'labels': labels}
    
    @classmethod
    def birnn_collate_batch(cls, batch):
        feats, labels = [], []
        for x, y in batch:
            feats.append(torch.tensor(x))
            labels.append(y)
        # 直接将一个ndarray的列表转化为张量是非常慢的，
        # 所以需要提前将列表转化为一整个ndarray
        feats = pad_sequence(feats, padding_value=0, batch_first=True)
        feats = torch.tensor(np.array(feats), dtype=torch.long)
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        return {'input_feats': feats, 'labels': labels}
    
    @classmethod
    def bert_collate_batch(cls, batch):
        feats, labels = [], []
        for x, y in batch:
            if len(x) > 512:
                continue
            feats.append(torch.tensor(x))
            labels.append(y)
        # 直接将一个ndarray的列表转化为张量是非常慢的，
        # 所以需要提前将列表转化为一整个ndarray
        feats = pad_sequence(feats, padding_value=0, batch_first=True)
        attn_mask = (feats != 0)
        feats = torch.tensor(np.array(feats), dtype=torch.long)
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        return {'input_feats': feats, 'labels': labels, 'attn_mask': attn_mask}

class IMDb:
    def __init__(
        self, 
        dataset_type: str,
    ):
        download_dir = download_imdb()
        data_dir = os.path.join(download_dir, 'train_valid_test')
        if os.path.exists(data_dir) == False:
            split_aclimdb_dataset(download_dir)
        valid_types = ['train', 'valid', 'test']
        assert dataset_type in valid_types, f"Expected 'train', 'valid', or 'test', but got '{dataset_type}'."
        self.dataset_type = dataset_type
        self.data_dir = os.path.join(data_dir, self.dataset_type)
        
        dataset_class = [cla for cla in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, cla))]
        dataset_class.sort()
        class_indices = dict((k, v) for v, k in enumerate(dataset_class))
        if self.dataset_type == 'train':
            json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            with open('class_indices.json', 'w') as json_file:
                json_file.write(json_str)
        self.class_indices = class_indices

        def read_imdb():
            """load IMDb dataset's sentences and labels"""  
            data, labels = [], []
            
            print(f'Reading {dataset_type} dataset...')
            
            for label, id in self.class_indices.items():
                folder_name = os.path.join(self.data_dir, label)
                for file in os.listdir(folder_name):
                    with open(os.path.join(folder_name, file), 'rb') as f:
                        review = f.read().decode('utf-8').replace('\n', '')
                        review = review.replace('<br />', '')
                        data.append(review)
                        labels.append(id)
            return data, labels
        
        self.data = read_imdb()
        print(f'{dataset_type} dataset size = ', len(self.data[0]))
        for x, y in zip(self.data[0][:3], self.data[1][:3]):
            print('label: ', y, 'review: ', x[0:60])
        
    def tokenize(self):
        self.tokens = []
        # print(pd.DataFrame([review for review in data[0]]).shape)
        # self.sentences = pd.DataFrame([sent_tokenize(review) for review in self.data[0]])

        def clean_text(str_in):
            clean_t = re.sub("[^A-Za-z]+", " ", str_in).strip().lower()
            return clean_t         
        
        def rem_sw(str_in):
            # nltk.download('word2vec_sample')
            sw = nltk.corpus.stopwords.words('english')
            filt = [word for word in str_in.split() if word not in sw]
            ex_text = ' '.join(filt)
            return ex_text
        
        def stem_fun(str_in, sw_in="stem"):
            if sw_in == "stem":
                stem = nltk.stem.PorterStemmer()
                stem_fun = [stem.stem(word) for word in str_in.split()]
            elif sw_in == "lemma":
                stem = nltk.stem.WordNetLemmatizer()
                stem_fun = [stem.lemmatize(word) for word in str_in.split()]
            stem_fun = ' '.join(stem_fun)
            return stem_fun

        print(f'Tokenize for {self.dataset_type} dataset')
        for review in tqdm(self.data[0]):
            review = rem_sw(clean_text(review))
            review = stem_fun(rem_sw(clean_text(review)), "lemma")
            self.tokens.append(word_tokenize(review))
            # break


    def build_vocab(self, min_freq=5, min_len=2, max_size=None):
        if self.dataset_type != 'train':
            print("Only train datset can be used to build vocab.")
            return 
        freqency = defaultdict(int)
        for tokens in self.tokens:
            for token in tokens:
                freqency[token] += 1
            # break

        print(
            f'unique tokens = {len(freqency)}, ' + 
            f'max freq = {max(freqency.values())}, ' + 
            f'min freq = {min(freqency.values())}'      
        )
        self.token2id = {}
        self.id2token = {}
        total_count = 0
        for token, freq in sorted(freqency.items(), key=lambda x: -x[1]):
            if max_size and len(self.token2id) >= max_size:
                break
            if freq > min_freq:
                if (min_len is None) or (min_len and len(token) >= min_len):
                    self.token2id[token] = len(self.token2id) + 1
                    self.id2token[len(self.token2id) + 1] = token
                    total_count += freq
            else:
                break
        print(
            f'min_freq = {min_freq}, min_len = {min_len}, ' + 
            f'max_size = {max_size}, ' + 
            f'remaining tokens = {len(self.token2id)}, ' + 
            f'in-vocab rate = {total_count / sum(freqency.values())}'
        )
    
    def convert_tokens_to_ids(self, token2id):
        self.token_ids = []
        print(f'Convert tokens to ids for {self.dataset_type} dataset')
        for tokens in tqdm(self.tokens):
            token_ids = []
            for token in tokens:
                if token in token2id:
                    token_ids.append(token2id[token])
            self.token_ids.append(token_ids)

    def write_ids_pickle(
            self,
            path_in=os.path.join('..', 'outputs'), 
            name_in="IMDb_token_ids",
        ):
        name_in = name_in + "_" + self.dataset_type
        pickle.dump(self.token_ids, open(os.path.join(path_in, name_in + ".pk"), 'wb'))
    
    def read_ids_pickle(
            self,
            path_in=os.path.join('..', 'outputs'), 
            name_in="IMDb_token_ids",
        ):
        name_in = name_in + "_" + self.dataset_type
        the_data_t = pickle.load(open(os.path.join(path_in, name_in + ".pk"), 'rb'))
        return the_data_t
    
    def visual(self):
        if self.dataset_type != 'train':
            print("Only visualize train datset.")
            return 
        # 生成词云
        all_words = ' '.join([text for review in train_data.tokens for text in review ])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.savefig('word_cloud.png', dpi=300, bbox_inches='tight')  # 保存为PNG格式
        plt.close()

        # 可视化特征重要性
        # 计算特征的重要性（例如，使用卡方检验）
        # chi_scores = chi2(X, df['vader_compound'] >= 0)  # 假设正向情感为重要特征

        # 将分数映射回词汇表
        # important_words = [(vectorizer.get_feature_names_out()[i], chi_scores[0][i]) for i in chi_scores[1].argsort()[:-11:-1]]
        # print("Top 10 important features:", important_words)
        # plt.figure(figsize=(10, 6))
        # sns.barplot(x=[score for word, score in important_words], y=[word for word, score in important_words])
        # plt.title('Feature Importance')
        # plt.xlabel('Importance Score')
        # plt.ylabel('Words')
        # plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')  # 保存为PNG格式
        # plt.close()

        print("图像已成功保存.")



if __name__ == "__main__":
    # nltk.download('punkt_tab')
    train_data, valid_data = IMDb(dataset_type='train'), IMDb(dataset_type='valid')
    
    train_data.tokenize()
    train_data.visual()
    # train_data.build_vocab(min_freq=3)
    # for ids, tokens in train_data.id2token.items():
    #     if tokens == 'br':
    #         print("Yes")
        # print(tokens)
    # train_data.convert_tokens_to_ids()
    # train_data.write_pickle()

    # valid_data.tokenize()
    # valid_data.build_vocab(min_freq=3)
    # valid_data.convert_tokens_to_ids()
    # valid_data.write_pickle()
    # print(len(train_data.class_indices))
    # train_token_ids = train_data.read_pickle()
    # valid_token_ids = valid_data.read_pickle()
    # print(pd.DataFrame(train_token_ids).shape,pd.DataFrame(valid_token_ids).shape)
    # print(train_token_ids[0], valid_token_ids[0])



