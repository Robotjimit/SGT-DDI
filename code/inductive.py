import os
import sys
from tqdm.auto import tqdm
import numpy as np
import os
import torch
import time
import time as t
from prettytable import PrettyTable
from load import DrugDataset, DrugDataLoader
from model import model
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')
from compute import  split_train_valid, train, test

class smile(object):
    def __init__(self, batch_size=100, epoches=50, num_workers=10):

        self.batch_size = batch_size
        self.epoches = epoches
        self.num_workers = num_workers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def main(self):
        data_size_ratio = 1
        neg_samples = 1
        base_path = '../data1/inductive/'
        seed= [42,35,51]
        key = 3
        task = 'multiclass'
        for fold in range(1,key):
            self.model_path = f'test/v2.pkl'

            df_ddi_train = pd.read_csv('../data1/inductive/42/train.csv')
            df_ddi_s1 = pd.read_csv('../data1/inductive/42/s1.csv')
            df_ddi_s2 = pd.read_csv('../data1/inductive/42/s2.csv')

            train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
            train_tup,val_tup = split_train_valid(train_tup, seed=13,test_ratio=0.05)
            s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
            s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]

#             all_id = train_data.get_drug_ids()
#             s1_id = s1_data.get_drug_ids()

            args = self.batch_size, self.epoches, self.num_workers, fold, self.model_path,task
            print(f"Training on {base_path}{fold} ")

            if os.path.exists(self.model_path):
                self.model = torch.load(self.model_path, map_location=self.device)
            else:
                self.model = model().to(self.device) #'../ckpt/pretrained_gt/checkpoints/model.pth'
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            print(self.model)

            with DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples,shuffle = True) as train_data, DrugDataset(val_tup,neg_ent=1,shuffle = True) as val_data:
                print(train_data.get_drug_ids())
                train(train_data, val_data, args, self.model, self.optimizer, self.device)

            # with DrugDataset(s2_tup, disjoint_split=True,shuffle = False) as s2_data:
            #     test(s2_data, args, self.model, self.device, 'result/s2.csv')
            with DrugDataset(s1_tup, disjoint_split=True,shuffle = False) as s1_data:
                test(s1_data, args, self.model, self.device, 'result/s1.csv')






if __name__ == "__main__":
    torch.cuda.empty_cache()
    smile = smile()
    smile.main()



