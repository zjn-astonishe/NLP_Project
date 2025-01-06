import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

def split_aclimdb_dataset(data_dir, test_size=0.5, random_state=None):
    """
    将 ACLIMDB 测试集平均划分为验证集和最终测试集，并显示进度条。
    
    参数:
        data_dir (str): 原始 ACLIMDB 测试集目录路径。
        test_size (float or int): 如果是浮点数，则表示测试集所占比例；如果是整数，则表示测试集样本数量。
        random_state (int): 随机种子，确保结果可重复。
        
    返回:
        None
    """
    
    # 确保输出目录存在
    output_dir = os.path.join(data_dir, 'train_valid_test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_output_dir = os.path.join(output_dir, 'train')
    val_output_dir = os.path.join(output_dir, 'valid')
    test_output_dir = os.path.join(output_dir, 'test')
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    
    os.makedirs(val_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    
    for sentiment in ['neg', 'pos']:
        sentiment_valid_test_path = os.path.join(test_dir, sentiment)
        sentiment_train_path = os.path.join(train_dir, sentiment)

        files_valid_test = [os.path.join(sentiment_valid_test_path, f) for f in os.listdir(sentiment_valid_test_path) if f.endswith('.txt')]
        files_train = [os.path.join(sentiment_train_path, f) for f in os.listdir(sentiment_train_path) if f.endswith('.txt')]
        # 使用 tqdm 显示读取文件的进度条
        print(f"Processing {sentiment} files...")
        files_val, files_test = train_test_split(files_valid_test, test_size=test_size, random_state=random_state)
        
        # 创建相应的子文件夹
        train_sentiment_dir = os.path.join(train_output_dir, sentiment)
        val_sentiment_dir = os.path.join(val_output_dir, sentiment)
        test_sentiment_dir = os.path.join(test_output_dir, sentiment)
        
        os.makedirs(train_sentiment_dir, exist_ok=True)
        os.makedirs(val_sentiment_dir, exist_ok=True)
        os.makedirs(test_sentiment_dir, exist_ok=True)
        
        print(f"Copying {sentiment} train files...")
        for file in tqdm(files_train, desc=f'Train {sentiment}', unit='file'):
            shutil.copy(file, train_sentiment_dir)
        
        # 使用 tqdm 显示复制文件的进度条
        print(f"Copying {sentiment} validation files...")
        for file in tqdm(files_val, desc=f'Validation {sentiment}', unit='file'):
            shutil.copy(file, val_sentiment_dir)
        
        print(f"Copying {sentiment} test files...")
        for file in tqdm(files_test, desc=f'Test {sentiment}', unit='file'):
            shutil.copy(file, test_sentiment_dir)
    
    print(f"Train set saved to: {train_output_dir}")
    print(f"Validation set saved to: {val_output_dir}")
    print(f"Test set saved to: {test_output_dir}")

# 使用示例
if __name__ == "__main__":
    aclimdb_test_dir = os.path.join('..', 'data', 'aclImdb')    # 替换为你的 ACLIMDB 测试集目录路径

    random_seed = 42  # 设置随机种子以保证结果可复现
    
    split_aclimdb_dataset(aclimdb_test_dir, test_size=0.5, random_state=random_seed)