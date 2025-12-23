import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from unimol_tools import UniMolRepr 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


clf = UniMolRepr(data_type='molecule', remove_hs=False)

df = pd.read_csv('data1/DDInter/drug_smiles_rdkit_valid.csv')
# 提取列表
id_list = df['DDInterID'].tolist()
smile_list = df['Smile'].tolist()

unimol_repr = clf.get_repr(smile_list)['cls_repr']  # shape: (N, D)

np.savez_compressed("data1/DDInter/data.npz", drug=np.array(id_list), unimol=np.array(unimol_repr))

# import pandas as pd

# # 读取 drug_smiles.csv
# df_drug = pd.read_csv('data1/DDInter/drug_smiles_rdkit_valid.csv')  # 包含 ['ID', 'Smile']

# # # 清洗 SMILES
# # df_drug_clean = df_drug[df_drug['Smile'].notna()]
# # df_drug_clean = df_drug_clean[df_drug_clean['Smile'].str.strip().str.lower() != 'none']
# # df_drug_clean = df_drug_clean[df_drug_clean['Smile'].str.strip() != '']

# # # 获取有效药物 ID 列表
# valid_ids = set(df_drug['DDInterID'])

# # # 保存清洗后的药物文件
# # df_drug_clean.to_csv('data1/DDInter/drug_smiles_filtered.csv', index=False)

# # 读取 ddinter_dataset.csv
# df_ddi = pd.read_csv('data1/DDInter/ddinter_dataset.csv')  # 假设包含 ['DDInterID_A', 'DDInterID_B', 'level_sort']

# # 过滤掉包含无效药物的行
# df_ddi_filtered = df_ddi[
#     df_ddi['DDInterID_A'].isin(valid_ids) & df_ddi['DDInterID_B'].isin(valid_ids)
# ]

# # 保存清洗后的 DDI 文件
# df_ddi_filtered.to_csv('data1/DDInter/ddinter_dataset_filtered.csv', index=False)

# # 打印日志
# print(f"原始药物数：{len(df_drug)}")
# # print(f"有效药物数：{len(df_drug_clean)}")
# print(f"原始DDI数：{len(df_ddi)}")
# print(f"保留的DDI数：{len(df_ddi_filtered)}")
# from rdkit import Chem
# import pandas as pd

# # 读取过滤后的药物文件
# df = pd.read_csv('data1/DDInter/drug_smiles_filtered.csv')  # 包含 ['ID', 'Smile']

# # 记录能成功解析的
# valid_ids = []
# valid_smiles = []

# # 记录失败的
# invalid_ids = []
# invalid_smiles = []

# # 遍历检查
# for idx, row in df.iterrows():
#     smile = row['Smile']
#     mol = Chem.MolFromSmiles(smile)
#     if mol is not None:
#         valid_ids.append(row['DDInterID'])
#         valid_smiles.append(smile)
#     else:
#         invalid_ids.append(row['DDInterID'])
#         invalid_smiles.append(smile)

# # 构建 DataFrame 保存
# df_valid = pd.DataFrame({'DDInterID': valid_ids, 'Smile': valid_smiles})
# df_invalid = pd.DataFrame({'DDInterID': invalid_ids, 'Smile': invalid_smiles})

# # 保存结果
# df_valid.to_csv('data1/DDInter/drug_smiles_rdkit_valid.csv', index=False)
# df_invalid.to_csv('data1/DDInter/drug_smiles_rdkit_invalid.csv', index=False)

# print(f'有效 SMILES 数: {len(df_valid)}')
# print(f'无效 SMILES 数: {len(df_invalid)}')
