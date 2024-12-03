import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer 

from torch.utils.data import Dataset

import torch.optim as optim
from tqdm import tqdm

import torch.nn as nn

from molformer import *

from finetune_script import SmilesDataset, load_model

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



config = SimpleNamespace(
    lr=1e-3,
    use_gpu=False,
    epochs=10,
    batch_size=1,
    hidden_dropout_prob=0.03,
    train_path='data/bio-chem-train.csv',
    dev_path='data/bio-chem-val.csv',
    test_path='data/bio-chem-test.csv',
    num_heads = 4,
    num_layers =4,
    dim = 256,
    max_seq_len = 80,
    vocab_size = 767,
    mutation_vocab_size = 7,
    mask = False
    )



device = torch.device('cpu')
print("Device:", device)

# Create dataloaders 

  
   
dataset =SmilesDataset(config.test_path)

print('Created datasets')

dataloader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size)

print('Created dataloaders')
   

model = MolformerModel(config)
model = model.to(device)

lr = config.lr
optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.01)


model, _ = load_model(model, optimizer, 'outputs/bio_1e-3/bio_model_backbone_finetuned.pt')
print('Loaded model:', 'outputs/bio_1e-3/bio_model_backbone_finetuned.pt')


hidden_states = []


for batch in tqdm(dataloader, desc=f'test'):

 

    model.eval()

    b_ids = batch['input_ids']
    b_logp = batch['logp']
    b_qed = batch['qed']
    b_sas = batch['sas']

    b_map3k1 = batch['expression_MAP3K1']
    b_pik3ca = batch['expression_PIK3CA']
    b_tp53 = batch['expression_TP53']
    b_mutation_idx = batch['mutation_idx']
          
          
    b_ids = b_ids.to(device)
    b_logp = b_logp.to(device)
    b_qed = b_qed.to(device)
    b_sas = b_sas.to(device)

    b_map3k1 = b_map3k1.to(device)
    b_pik3ca = b_pik3ca.to(device)
    b_tp53 = b_tp53.to(device)
    b_mutation_idx = b_mutation_idx.to(device)


         
    logits, pred_logp, pred_qed, pred_sas, init_hidden_state = model(b_ids ,pad_indx = 1)

    init_hidden_state = init_hidden_state.reshape(-1, init_hidden_state.size(-1)) # seq_len, dim

   

    hidden_states.append(init_hidden_state.detach().numpy())



hidden_states = [arr.flatten() for arr in hidden_states] # (80*256)


hidden_states = np.stack(hidden_states, axis=0) # (n, 80*256)



## Make TSNE visualization

tsne = TSNE(n_components =2, random_state = 7)
tsne_state = tsne.fit_transform(hidden_states)




plt.figure(figsize=(8,6))
plt.scatter(tsne_state[:,0], tsne_state[:,1], s=10, cmap='viridis')
plt.title('t-SNE visualization of molecular representations')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()







