import torch
from torch.utils.data import Dataset
import numpy as np
from smiles_encoder import SmilesEncoder
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch


import rdkit.Chem as Chem
import torch
import random
import numpy as np
import pandas as pd
# import pyximport
# pyximport.install(setup_args={"include_dirs": np.get_include()})

from dgllife.utils import smiles_to_complete_graph
import torch
from rdkit import Chem
from dgllife.utils import smiles_to_complete_graph
from functools import partial
from torch_geometric.data import Data, Batch
import random
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from dgl.data.utils import load_graphs

import random
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from dgllife.utils import smiles_to_complete_graph
import dgl
from dgl.data.utils import save_graphs, load_graphs
from functools import partial
from rdkit import Chem
from sklearn.preprocessing import LabelEncoder
from dgl import shortest_dist

def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(atom.GetAtomicNum())
    return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

def featurize_edges(mol, add_self_loop=False):
    feats = []
    num_atoms = mol.GetNumAtoms()
    atoms = list(mol.GetAtoms())
    distance_matrix = Chem.GetDistanceMatrix(mol)
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j or add_self_loop:
                feats.append(float(distance_matrix[i, j]))
    return {'dist': torch.tensor(feats).reshape(-1, 1).float()}

with np.load('../data1/data.npz') as data:
    # 假设data中包含drug_id, fps, unimol
    drug_ids = data['drug']
    fps = data['fps']
    unimols = data['unimol']

df_drugs_smiles = pd.read_csv('../data1/drugbank/drug_smiles.csv')
drug_id_mol_graph = {}
i = 0
for drug_id in drug_ids:
    # 获取对应的 SMILES 序列
    smiles = df_drugs_smiles.loc[df_drugs_smiles['drug_id'] == drug_id, 'smiles'].iloc[0]
    # 将 graph 添加到字典中
    # graph = smiles_to_complete_graph(
    #     smiles, add_self_loop=True, node_featurizer=featurize_atoms,
    #     edge_featurizer=partial(featurize_edges, add_self_loop=True))
    #
    # spd, path = shortest_dist(graph, root=None, return_paths=True)
    # graph.ndata["spd"] = spd
    # graph.ndata["path"] = path
    # # print(graph)
    drug_id_mol_graph[drug_id] = (unimols[i], smiles)
    i += 1

######################## Define the dataset ########################

class DrugDataset(Dataset):
    def __init__(self, tri_list, ratio=1.0,  neg_ent=1, disjoint_split=True, shuffle=True):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the dataset
        '''
        self.neg_ent = neg_ent
        self.tri_list = []
        self.ratio = ratio

        for h, t, r, *_ in tri_list:
            if ((h in drug_id_mol_graph) and (t in drug_id_mol_graph)):
                self.tri_list.append((h, t, r))
        if disjoint_split:
            d1, d2, *_ = zip(*self.tri_list)
            self.drug_ids = np.array(list(set(d1 + d2)))


        self.drug_ids = np.array([id for id in self.drug_ids if id in drug_id_mol_graph])

        if shuffle:
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]


    def get_drug_ids(self):
        """
        Returns all drug IDs in the dataset.
        """
        return self.drug_ids


    def read_smiles(self,smile):
        import os
        import csv
        import math
        import time
        import random
        import numpy as np

        import torch
        import torch.nn.functional as F
        from torch.utils.data.sampler import SubsetRandomSampler

        from torch_scatter import scatter
        from torch_geometric.data import Data, Dataset, DataLoader

        import rdkit
        from rdkit import Chem
        from rdkit.Chem.rdchem import HybridizationType
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit.Chem import AllChem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')

        ATOM_LIST = list(range(1, 119))
        CHIRALITY_LIST = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]
        BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
        BONDDIR_LIST = [
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ]
        mol = Chem.MolFromSmiles(smile)

        # mol = Chem.AddHs(mol)
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data


    def collate_fn(self, batch):

        pos_rels = []
        pos_h_graph_samples = []
        pos_h_unimol_samples = []
        pos_t_graph_samples = []
        pos_t_unimol_samples = []

        label=[]

        for h, t, r in batch:
            pos_rels.append(r)

            h_unimol, h_graph = drug_id_mol_graph[h]
            t_unimol, t_graph = drug_id_mol_graph[t]

            pos_h_unimol_samples.append(h_unimol)
            pos_h_graph_samples.append(h_graph)
            pos_t_unimol_samples.append(t_unimol)
            pos_t_graph_samples.append(t_graph)

        pos_h_unimol_samples = [torch.from_numpy(np_array) for np_array in pos_h_unimol_samples]
        pos_h_graph = [self.read_smiles(smiles) for smiles in pos_h_graph_samples]
        pos_t_unimol_samples = [torch.from_numpy(np_array) for np_array in pos_t_unimol_samples]# pos_t_graph_samples = self.graph_process(pos_t_graph_samples)
        pos_t_graph = [self.read_smiles(smiles) for smiles in pos_t_graph_samples]

        h_graph_samples = pos_h_graph
        t_graph_samples = pos_t_graph
        h_unimol_samples = pos_h_unimol_samples
        t_unimol_samples = pos_t_unimol_samples
        label = pos_rels

        h_unimol_samples = torch.stack(h_unimol_samples, dim=0)
        t_unimol_samples = torch.stack(t_unimol_samples, dim=0)
        h_graph_samples = Batch.from_data_list(h_graph_samples)
        t_graph_samples = Batch.from_data_list(t_graph_samples)
        tri = (h_unimol_samples, h_graph_samples, t_unimol_samples, t_graph_samples)

        return tri,label


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
class DrugDataLoader(DataLoader):
    def __init__(self, data, task=None, **kwargs):
        # # 如果没有传入collate_fn，就使用data的默认collate_fn
        # collate_fn = data.multiclass_collate_fn if task is not None else data.collate_fn
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

