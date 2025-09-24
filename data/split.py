import pandas as pd
from sklearn.model_selection import train_test_split

def generate_datasets(ddi_filepath: str):

    # 读取 DDI 数据
    ddi_df = pd.read_csv(ddi_filepath)
    # 五折划分

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X = ddi_df.drop('type', axis=1)
    y = ddi_df['type']

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(X)):
        fold_dir = f"fold{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)
        # test集
        train_val = ddi_df.iloc[train_val_idx]
        test = ddi_df.iloc[test_idx]
        # train/val 6:2
        train, val = train_test_split(train_val, test_size=0.25, random_state=fold_idx)
        # 保存
        train.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        test.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

# 在主函数中调用
if __name__ == '__main__':
    # ... existing code ...
    generate_datasets('ddis.csv')  # 替换为实际路径
    # ... existing code ...
