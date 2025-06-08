# SGT-DDI: A multimodal information integration Framework for Robust Prediction of Drug-Drug Interactions

## File list

- compute.py: The Folder contains the trained model of DMFF-DTA.
- load.py: The code for data preprocessing.
- model.py: The code about model.
- transductive.py: The code about transductive setting.
- inductive.py: The code about inductive setting.

## Dataset

Before training, you can unzip data.tar.gz to obtain the data required for training.

## Run Code

### Step 1: unzip data.tar.gz

### Step 2: run transductive.py/inductive.py

## Requirements

- networkx==3.1
- numpy==1.24.3
- pandas==1.5.3
- rdkit==2022.03.2
- scikit_learn==1.3.0
- scipy==1.10.1
- torch==1.12.1
- torch_geometric==2.3.1
- tqdm==4.65.0
