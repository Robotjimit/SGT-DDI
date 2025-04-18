
import os
import sys
from tqdm.auto import tqdm
import numpy as np
import os
import torch

from load import DrugDataset, DrugDataLoader
from model import model
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')
from compute import split_train_valid, train, test

class smile(object):
    def __init__(self, batch_size=256, epoches=400, num_workers=12):

        self.batch_size = batch_size
        self.epoches = epoches
        self.num_workers = num_workers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def main(self):

        base_path = '../data1/drugbank/fold'
        task = 'multiclass'

        fold = 0


        self.model_path = None

        ddi = pd.read_csv('../data1/drugbank/ddis.csv')
        train_tup = [(h, t, r) for h, t, r in zip(ddi['d1'], ddi['d2'], ddi['type'])]
        train_tup, val_tup,test_tup = split_train_valid(train_tup, 42, val_ratio=0.1,test_ratio=0.1)


        args = self.batch_size, self.epoches, self.num_workers, fold, self.model_path,task
        print(f"Training on {base_path} ddi.csv with train size of {len(train_tup)} , val size of {len(val_tup)}, test size of {len(test_tup)}")
        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path, map_location=self.device)
        else:
            self.model = model().to(self.device)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)


        with DrugDataset(train_tup) as train_data, DrugDataset(val_tup) as val_data:
            train(train_data, val_data, args, self.model, self.optimizer, self.device)

        with DrugDataset(test_tup) as test_data:
            test(test_data, args, self.model, self.device, 'result/trans_test.csv')





if __name__ == "__main__":
    torch.cuda.empty_cache()
    smile = smile()
    smile.main()
