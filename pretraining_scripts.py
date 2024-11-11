

import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer 
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset

import torch.optim as optim
from tqdm import tqdm

import torch.nn as nn

from molformer import *

import os





## Load and tokenize data

## Vocab size: 767
## Start token: 0
## End token: 2
## Pad token: 1

class SmilesDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='seyonec/ChemBERTa-zinc-base-v1', max_length=80):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Smile strings and scores
        smiles = self.data.iloc[idx]['smiles']
        logp = self.data.iloc[idx]['logP']
        qed = self.data.iloc[idx]['qed']
        sas = self.data.iloc[idx]['SAS']

        # Tokenize smiles string

        tokens = self.tokenizer(smiles, padding = 'max_length', max_length = self.max_length, truncation = True, return_tensors = 'pt')

        input_ids = tokens['input_ids'].squeeze(0) # Remove batch dim

        logp_tensor = torch.tensor(logp, dtype = torch.float32)
        qed_tensor = torch.tensor(qed, dtype = torch.float32)
        sas_tensor = torch.tensor(sas, dtype = torch.float32)
       
       

        return {
            'input_ids': input_ids,
            'logp': logp_tensor,
            'qed': qed_tensor,
            'sas': sas_tensor
        }








def save_model(model, optimizer, args, config, filepath):
    # Taken from CS224N Project spec
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")



def load_model(model, optimizer, filepath):
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])

    return model,optimizer


 



def model_eval(dataloader, model, device, args):
    model.eval() 

    val_loss = 0
    num_batches = 0

    dev_ce_loss = 0
    dev_logp_loss = 0
    dev_qed_loss = 0
    dev_sas_loss = 0

    for batch in tqdm(dataloader, desc=f'dev'):

        b_ids = batch['input_ids']
        b_logp = batch['logp']
        b_qed = batch['qed']
        b_sas = batch['sas']
          
        b_ids = b_ids.to(device)
        b_logp = b_logp.to(device)
        b_qed = b_qed.to(device)
        b_sas = b_sas.to(device)

        
        logits, pred_logp, pred_qed, pred_sas = model(b_ids,  pad_indx = 1)

        ce_loss = F.cross_entropy(logits.permute(0, 2, 1), b_ids, reduction='sum') / args.batch_size

        logp_loss = F.mse_loss(pred_logp, b_logp, reduction='mean')
        qed_loss = F.mse_loss(pred_qed, b_qed,reduction='mean')
        sas_loss = F.mse_loss(pred_sas, b_sas,reduction='mean')


        loss = args.beta*ce_loss + args.phi*logp_loss + args.gamma*qed_loss + args.delta*sas_loss

           

        val_loss += loss.item()

        dev_ce_loss += args.beta*ce_loss.item()
        dev_logp_loss += args.phi*logp_loss.item()
        dev_qed_loss += args.gamma*qed_loss.item()
        dev_sas_loss += args.delta*sas_loss.item()


        num_batches += 1

    val_loss = val_loss / (num_batches)

    dev_ce_loss = dev_ce_loss / (num_batches)
    dev_logp_loss = dev_logp_loss / (num_batches)
    dev_qed_loss = dev_qed_loss / (num_batches)
    dev_sas_loss = dev_sas_loss / (num_batches)

    

    return val_loss, dev_ce_loss, dev_logp_loss, dev_qed_loss, dev_sas_loss




def evaluate(args, config):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    print("Args:", args.use_gpu)
    print("Device:", device)

    test_dataset = SmilesDataset(config.test_path)

    print('Created datasets')

    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size,
                                )

    print('Created dataloaders')

    model = MolformerModel(config)
    model = model.to(device)


    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.01)

    model, _ = load_model(model, optimizer, args.saved_model_filepath)
    print('Loaded model:', args.saved_model_filepath)

 

    model.eval() 

    pred_logps = []
    pred_qeds = []
    pred_sass = []

    true_logps = []
    true_qeds = []
    true_sass = []

    for batch in tqdm(test_dataloader, desc=f'dev'):

    


        b_ids = batch['input_ids']
        b_logp = batch['logp']
        b_qed = batch['qed']
        b_sas = batch['sas']
          
        b_ids = b_ids.to(device)
        b_logp = b_logp.to(device)
        b_qed = b_qed.to(device)
        b_sas = b_sas.to(device)

        
        logits, pred_logp, pred_qed, pred_sas = model(b_ids,  pad_indx = 1)

        

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

    res.to_csv(args.filepath_inference, index=False)





def train(args, config):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    print("Args:", args.use_gpu)
    print("Device:", device)

    # Create dataloaders 

  
   
    train_dataset =SmilesDataset(config.train_path)
    dev_dataset = SmilesDataset(config.dev_path)

    print('Created datasets')

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                )
    
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              )


    print('Created dataloaders')
   

    model = MolformerModel(config)
    model = model.to(device)

    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    best_dev_loss = float('inf')


    train_losses = []
    train_ce_losses = []
    train_logp_losses = []
    train_qed_losses = []
    train_sas_losses = []
    
    dev_losses = []
    dev_ce_losses = []
    dev_logp_losses = []
    dev_qed_losses = []
    dev_sas_losses = []

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        train_ce_loss = 0
        train_logp_loss = 0
        train_qed_loss = 0
        train_sas_loss = 0

        for batch in tqdm(train_dataloader, desc=f'train-{epoch}'):

            
            b_ids = batch['input_ids']
            b_logp = batch['logp']
            b_qed = batch['qed']
            b_sas = batch['sas']
          
            b_ids = b_ids.to(device)
            b_logp = b_logp.to(device)
            b_qed = b_qed.to(device)
            b_sas = b_sas.to(device)

            optimizer.zero_grad()
            logits, pred_logp, pred_qed, pred_sas = model(b_ids, pad_indx = 1)


            ce_loss = F.cross_entropy(logits.permute(0, 2, 1), b_ids, reduction='sum') / args.batch_size


            logp_loss = F.mse_loss(pred_logp, b_logp, reduction='mean')
            qed_loss = F.mse_loss(pred_qed, b_qed,reduction='mean')
            sas_loss = F.mse_loss(pred_sas, b_sas,reduction='mean')

            loss = args.beta*ce_loss + args.phi*logp_loss + args.gamma*qed_loss + args.delta*sas_loss

            if num_batches % 100 == 0:
                print(f"CE Loss {ce_loss}: logp loss :: {logp_loss :.3f}, qed loss :: {qed_loss :.3f}, sas loss :: {sas_loss :.3f}")

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_ce_loss += args.beta*ce_loss.item()
            train_logp_loss += args.phi*logp_loss.item()
            train_qed_loss += args.gamma*qed_loss.item()
            train_sas_loss += args.delta*sas_loss.item()

            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_ce_loss = train_ce_loss / (num_batches)
        train_logp_loss = train_logp_loss / (num_batches)
        train_qed_loss = train_qed_loss / (num_batches)
        train_sas_loss = train_sas_loss / (num_batches)

    
        dev_loss, dev_ce_loss, dev_logp_loss, dev_qed_loss, dev_sas_loss = model_eval(dev_dataloader, model, device, args)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print('Saving model')
            if os.path.exists(args.filepath_mod):
                os.remove(args.filepath_mod)
            save_model(model, optimizer, args, config, args.filepath_mod)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}")

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        train_ce_losses.append(train_ce_loss)
        train_logp_losses.append(train_logp_loss)
        train_qed_losses.append(train_qed_loss)
        train_sas_losses.append(train_sas_loss)

        dev_ce_losses.append(dev_ce_loss)
        dev_logp_losses.append(dev_logp_loss)
        dev_qed_losses.append(dev_qed_loss)
        dev_sas_losses.append(dev_sas_loss)

    losses_df = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss' : dev_losses,
        'train_ce': train_ce_losses,
        'train_logp': train_logp_losses,
        'train_qed': train_qed_losses,
        'train_sas_loss': train_sas_losses,
        'val_ce': dev_ce_losses,
        'val_logp': dev_logp_losses,
        'val_qed': dev_qed_losses,
        'val_sas_loss': dev_sas_losses
    })

    print('saving losses')

    

    losses_df.to_csv(args.filepath_losses)


