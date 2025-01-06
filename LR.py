import numpy as np
import torch
import os

from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from tfidf import TFIDF, visualize_feature_importance
from dataset import IMDb, myDataset, DataCollator
from model import LR


def train_one_epoch(model, data_loader, device, epoch):
    model.train()
    epoch_loss = []

    with tqdm(data_loader, desc=f"Epoch {epoch}", unit="batch") as tepoch:
        for batch in tepoch:
            inputs = batch['input_feats'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs, labels)

            loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            tepoch.set_postfix(loss=loss.item())
    avg_epoch_loss = np.mean(epoch_loss)
    print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    epoch_loss = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            inputs = batch['input_feats'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(inputs, labels)
            loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
            logits = outputs.logits if isinstance(outputs, dict) else outputs[1]

            # 记录损失值
            epoch_loss.append(loss.item())

            # 获取预测标签
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_predictions.extend(preds)
            all_labels.extend(labels)
    
    # 计算平均损失和准确率
    avg_loss = np.mean(epoch_loss)
    accuracy = accuracy_score(all_labels, all_predictions)

    # 打印分类报告（可选）
    # print("\nClassification Report:")
    # print(classification_report(all_labels, all_predictions))

    return {
        "avg_loss": avg_loss,
        "accuracy": accuracy
    }

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

if __name__ == "__main__":
    # 加载并预处理数据集
    train_data, valid_data, test_data = IMDb(dataset_type='train'), IMDb(dataset_type='valid'), IMDb(dataset_type='test')
    os.makedirs(os.path.join('..', 'outputs'), exist_ok=True)
    
    train_data.tokenize()
    train_data.build_vocab(min_freq=3)
    train_data.convert_tokens_to_ids(train_data.token2id)
    train_data.write_ids_pickle()
    
    valid_data.tokenize()
    valid_data.convert_tokens_to_ids(train_data.token2id)
    valid_data.write_ids_pickle()
    
    test_data.tokenize()
    test_data.convert_tokens_to_ids(train_data.token2id)
    test_data.write_ids_pickle()


    train_token_ids = train_data.read_ids_pickle()
    valid_token_ids = valid_data.read_ids_pickle()
    test_token_ids = test_data.read_ids_pickle()

    train_max_ids = 0
    train_X, train_Y = [], []
    for data in tqdm(train_token_ids, desc="Processing train dataset's token ids"):
        if max(data) > train_max_ids:
            train_max_ids = max(data)
        train_X.append(data)
    train_Y = train_data.data[1]

    valid_X, valid_Y = [], []
    for data in tqdm(valid_token_ids, desc="Processing valid dataset's token ids"):
        valid_X.append(data)
    valid_Y = valid_data.data[1]

    test_X, test_Y = [], []
    for data in tqdm(test_token_ids, desc="Processing test dataset's token ids"):
        test_X.append(data)
    test_Y = test_data.data[1]

    # 特征提取
    # tfidf = TFIDF(train_max_ids)
    tfidf = TFIDF(train_max_ids, token2id=train_data.token2id)
    tfidf.fit(train_X)
    train_F = tfidf.transform(train_X)
    valid_F = tfidf.transform(valid_X)
    test_F = tfidf.transform(test_X)

    # print(pd.DataFrame(train_Y).shape, pd.DataFrame(valid_Y).shape)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置训练超参数和优化器
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0

    print(train_max_ids)
    # 初始化模型并移动到指定设备
    model = LR(train_max_ids, len(train_data.class_indices)).to(device)

    model.apply(init_weights)

    # 创建数据集和数据加载器
    train_dataset = myDataset(train_F, train_Y)
    valid_dataset = myDataset(valid_F, valid_Y)
    test_dataset = myDataset(test_F, test_Y)
    data_collator = DataCollator()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator.lr_collate_batch
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator.lr_collate_batch
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator.lr_collate_batch
    )

    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 模型训练
    best_val_accuracy = 0.0
    best_model_weights = None
    os.makedirs(os.path.join('.', 'weights'), exist_ok=True)
    for epoch in range(epochs):

        train_loss = train_one_epoch(model, train_dataloader, device, epoch)
        
        val_results = evaluate(model, valid_dataloader, device)
        val_accuracy = val_results['accuracy']

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, "./weights/lr_best_model.pth")
            print(f"New best validation accuracy: {best_val_accuracy:.4f}. Model saved.")
        
        torch.save(model.state_dict(), "./weights/lr_latest_model.pth")

    # 在训练结束后调用可视化函数
    visualize_feature_importance(tfidf, train_F, train_Y)

    # 加载最佳模型参数
    model.load_state_dict(torch.load("./weights/lr_best_model.pth"))
    print("Training complete. Loaded best model weights.")

    # 最终测试集评估
    test_results = evaluate(model, test_dataloader, device)
    print(f"Final Test results - Average Loss: {test_results['avg_loss']:.4f}, Accuracy: {test_results['accuracy']:.4f}")