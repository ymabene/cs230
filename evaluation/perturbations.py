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
from biomolformer import *

from finetune_script import SmilesDataset, load_model

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


## Script to generate perturbations in BioMolformer latent space

config = SimpleNamespace(
    lr=1e-3,
    use_gpu=False,
    epochs=10,
    batch_size=1,
    hidden_dropout_prob=0.03,
    train_path='data/bio-chem-train.csv',
    dev_path='data/bio-chem-val.csv',
    test_path='data/bio-chem-test.csv',
    mol_path = 'data/bio-chem-all-pathways-activated.csv',
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

## 
  
   
dataset =SmilesDataset(config.mol_path)

print('Created datasets')

dataloader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size)

print('Created dataloaders')
   

model = MolformerModel(config)
model = model.to(device)

bio_model = BIOMolformerModel(config)
bio_model = bio_model.to(device)


lr = config.lr
optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.01)


model, _ = load_model(model, optimizer, 'outputs/bio_1e-3/bio_model_backbone_finetuned.pt')
print('Loaded model:', 'outputs/bio_1e-3/bio_model_backbone_finetuned.pt')

bio_model, _ = load_model(bio_model, optimizer,'outputs/bio_1e-3/bio_model_finetuned.pt')
print('Loaded bio model:', 'outputs/bio_1e-3/bio_model_finetuned.pt')



pred_logps = []
pred_qeds = []
pred_sass = []
   
true_logps = []
true_qeds = []
true_sass = []


for batch in tqdm(dataloader, desc=f'mol'):

 

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

    # Perturb hidden state

    pert = (torch.rand(init_hidden_state.shape[0],init_hidden_state.shape[1],init_hidden_state.shape[2]) - .5) *.2

    hidden_state = init_hidden_state + pert

    hidden_state, pred_logp,  pred_qed, pred_sas, pred_map3k1, pred_pik3ca, pred_tp53  = bio_model(hidden_state, b_mutation_idx)



    pred_logps.extend(pred_logp.detach().numpy().tolist())
    true_logps.extend(b_logp.numpy().tolist())

    pred_qeds.extend(pred_qed.detach().numpy().tolist())
    true_qeds.extend(b_qed.numpy().tolist())

    pred_sass.extend(pred_sas.detach().numpy().tolist())
    true_sass.extend(b_sas.numpy().tolist())



    
    res = pd.DataFrame({
        'pred_logp' : pred_logps,
        'true_logp' : true_logps,
        'pred_qed': pred_qeds,
        'true_qed': true_qeds,
        'pred_sass': pred_sass,
        'true_sass': true_sass,    
    })

    res.to_csv('outputs/molecule_generation_results.csv', index=False)




 









