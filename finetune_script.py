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
from biomolformer import *


import os


## This file includes training (finetuning) scripts for BIOMolformer model



"""
Dataset class for SMILES molecular data

Parameters:
csv_file: Path to data
tokenizer_name: SMILES tokenizer
max_length: Maximum sequence length
"""
class SmilesDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='seyonec/ChemBERTa-zinc-base-v1', max_length=80):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.mutation_mapping = {'100':0, '010':1, '001':2, '110':3, '101':4, '011':5, '111':6}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Smile strings and scores
        smiles = self.data.iloc[idx]['SMILES']
        logp = self.data.iloc[idx]['LogP']
        qed = self.data.iloc[idx]['QED']
        sas = self.data.iloc[idx]['sas']
        map3k1 = self.data.iloc[idx]['expression_MAP3K1']
        pik3ca = self.data.iloc[idx]['expression_PIK3CA']
        tp53 = self.data.iloc[idx]['expression_TP53']
        map3k1_idx = int(self.data.iloc[idx]['mutation_MAP3K1'])
        pik3ca_idx = int(self.data.iloc[idx]['mutation_PIK3CA'])
        tp53_idx = int(self.data.iloc[idx]['mutation_TP53'])

        mutation_idx = self.mutation_mapping[str(map3k1_idx) + str(pik3ca_idx) + str(tp53_idx)]

        # Tokenize smiles string

        tokens = self.tokenizer(smiles, padding = 'max_length', max_length = self.max_length, truncation = True, return_tensors = 'pt')

        input_ids = tokens['input_ids'].squeeze(0) # Remove batch dim

        logp_tensor = torch.tensor(logp, dtype = torch.float32)
        qed_tensor = torch.tensor(qed, dtype = torch.float32)
        sas_tensor = torch.tensor(sas, dtype = torch.float32)

        map3k1_tensor = torch.tensor(map3k1, dtype = torch.float32)
        pik3ca_tensor = torch.tensor(pik3ca, dtype = torch.float32)
        tp53_tensor = torch.tensor(tp53, dtype = torch.float32)
        mutation_tensor = torch.tensor(mutation_idx, dtype = torch.int64)
       
       

        return {
            'input_ids': input_ids,
            'logp': logp_tensor,
            'qed': qed_tensor,
            'sas': sas_tensor,
            'expression_MAP3K1': map3k1_tensor,
            'expression_PIK3CA':pik3ca_tensor,
            'expression_TP53':tp53_tensor,
            'mutation_idx':mutation_tensor
        }







"""
Saves trained model.

Parameters:
model: Model to save
optimizer: Optimizer used during training
args: Arguments
config: Model Configurations
filepath: Path to save model
"""
def save_model(model, optimizer, args, config, filepath):
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


"""
Loads model for inference or further training.

Parameters:
model: Model to load
optimizer: Optimizer used for training model
filepath: Path where model is read from
"""
def load_model(model, optimizer, filepath):
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])

    return model,optimizer


 


"""
Computes validation loss of model during training.

Parameters:
dataloader: Validation dataloader
model: Backbone model used during training
bio_model: Model with classification heads for biological data
args: Training arguments
"""
def model_evaluate_bio(dataloader, model,bio_model, device, args):
    model.eval() 
    bio_model.eval()

    val_loss = 0
    num_batches = 0

    dev_ce_loss = 0
    dev_logp_loss = 0
    dev_qed_loss = 0
    dev_sas_loss = 0

    dev_map3k1_loss = 0
    dev_pik3ca_loss = 0
    dev_tp53_loss = 0

    for batch in tqdm(dataloader, desc=f'dev'):

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

          
        hidden_state, pred_logp,  pred_qed, pred_sas, pred_map3k1, pred_pik3ca, pred_tp53  = bio_model(init_hidden_state, b_mutation_idx)

        ce_loss = F.cross_entropy(logits.permute(0, 2, 1), b_ids, reduction='sum') / args.batch_size

        logp_loss = F.mse_loss(pred_logp, b_logp, reduction='mean')
        qed_loss = F.mse_loss(pred_qed, b_qed,reduction='mean')
        sas_loss = F.mse_loss(pred_sas, b_sas,reduction='mean')

        map3k1_loss = F.mse_loss(pred_map3k1, b_map3k1,reduction='mean')
        pik3ca_loss = F.mse_loss(pred_pik3ca, b_pik3ca,reduction='mean')
        tp53_loss = F.mse_loss(pred_tp53, b_tp53,reduction='mean')

        loss =  args.beta * ce_loss + args.phi*logp_loss + args.gamma*qed_loss + args.delta*sas_loss + args.theta *map3k1_loss + args.theta *pik3ca_loss + args.theta *tp53_loss



        val_loss += loss.item()

        dev_ce_loss += args.beta*ce_loss.item()
        dev_logp_loss += args.phi*logp_loss.item()
        dev_qed_loss += args.gamma*qed_loss.item()
        dev_sas_loss += args.delta*sas_loss.item()

        dev_map3k1_loss += args.theta *map3k1_loss.item()
        dev_pik3ca_loss += args.theta *pik3ca_loss.item()
        dev_tp53_loss +=  args.theta *tp53_loss.item()



        num_batches += 1

    val_loss = val_loss / (num_batches)

    dev_ce_loss = dev_ce_loss / num_batches
    dev_logp_loss = dev_logp_loss / (num_batches)
    dev_qed_loss = dev_qed_loss / (num_batches)
    dev_sas_loss = dev_sas_loss / (num_batches)

    dev_map3k1_loss = dev_map3k1_loss / num_batches
    dev_pik3ca_loss = dev_pik3ca_loss / num_batches
    dev_tp53_loss = dev_tp53_loss / num_batches


    

    return val_loss, dev_ce_loss, dev_logp_loss, dev_qed_loss, dev_sas_loss, dev_map3k1_loss, dev_pik3ca_loss, dev_tp53_loss



"""
Evaluates model during training and stores loss values and predicted values from model into a csv file.

Parameters:
args: Model arguments
config: Model configurations
"""
def evaluate(args, config):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    print("Args:", args.use_gpu)
    print("Device:", device)

    test_dataset = SmilesDataset(config.test_path)

    print('Created datasets')

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                )

    print('Created dataloaders')

    model = MolformerModel(config)
    model = model.to(device)

    bio_model = BIOMolformerModel(config)
    bio_model = bio_model.to(device)


    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.01)

    model, _ = load_model(model, optimizer, args.saved_model_filepath)
    print('Loaded model:', args.saved_model_filepath)

    bio_model, _ = load_model(bio_model, optimizer, args.saved_bio_model_filepath)
    print('Loaded bio model:', args.saved_bio_model_filepath)


    model.eval()
    bio_model.eval() 

    pred_logps = []
    pred_qeds = []
    pred_sass = []
    pred_map3k1s = []
    pred_pik3cas = []
    pred_tp53s = []

    true_logps = []
    true_qeds = []
    true_sass = []
    true_map3k1s = []
    true_pik3cas = []
    true_tp53s = []

    mutation_idxs = []

    for batch in tqdm(test_dataloader, desc=f'dev'):


    


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

        hidden_state, pred_logp,  pred_qed, pred_sas, pred_map3k1, pred_pik3ca, pred_tp53  = bio_model(init_hidden_state, b_mutation_idx)

        

        pred_logps.extend(pred_logp.detach().numpy().tolist())
        true_logps.extend(b_logp.numpy().tolist())

        pred_qeds.extend(pred_qed.detach().numpy().tolist())
        true_qeds.extend(b_qed.numpy().tolist())

        pred_sass.extend(pred_sas.detach().numpy().tolist())
        true_sass.extend(b_sas.numpy().tolist())

        pred_map3k1s.extend(pred_map3k1.detach().numpy().tolist())
        true_map3k1s.extend(b_map3k1.numpy().tolist())

        pred_pik3cas.extend(pred_pik3ca.detach().numpy().tolist())
        true_pik3cas.extend(b_pik3ca.numpy().tolist())

        pred_tp53s.extend(pred_tp53.detach().numpy().tolist())
        true_tp53s.extend(b_tp53.numpy().tolist())

        mutation_idxs.extend(b_mutation_idx.detach().numpy().tolist())


    res = pd.DataFrame({
        'pred_logp' : pred_logps,
        'true_logp' : true_logps,
        'pred_qed': pred_qeds,
        'true_qed': true_qeds,
        'pred_sass': pred_sass,
        'true_sass': true_sass,
        'pred_map3k1s': pred_map3k1s,
        'true_map3k1s': true_map3k1s,
        'pred_pik3cas': pred_pik3cas,
        'true_pik3cas': true_pik3cas,
        'pred_tp53s': pred_tp53s,
        'true_tp53s': true_tp53s,
        'mutation_idx': mutation_idxs,
       
    })

    res.to_csv(args.filepath_inference, index=False)




"""
Trains model and stores loss values and predicted values from model into a csv file.

Parameters:
args: Model arguments
config: Model configurations
"""
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


    model, _ = load_model(model, optimizer, args.saved_model_filepath)
    print('Loaded model:', args.saved_model_filepath)


    bio_model = BIOMolformerModel(config)
    bio_model = bio_model.to(device)

    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    best_dev_loss = float('inf')


    train_losses = []
   
    train_ce_losses = []
    train_logp_losses = []
    train_qed_losses = []
    train_sas_losses = []
    train_map3k1_losses = []
    train_pik3ca_losses = []
    train_tp53_losses = []
    
    dev_losses = []
  
    dev_ce_losses = []
    dev_logp_losses = []
    dev_qed_losses = []
    dev_sas_losses = []

    dev_map3k1_losses = []
    dev_pik3ca_losses = []
    dev_tp53_losses = []


    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        
        #model.eval()
        train_loss = 0
        num_batches = 0

        train_ce_loss = 0

        train_logp_loss = 0
        train_qed_loss = 0
        train_sas_loss = 0

        train_map3k1_loss = 0
        train_pik3ca_loss = 0
        train_tp53_loss = 0

      
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}'):

        
            
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

         
            optimizer.zero_grad()

            hidden_state, pred_logp,  pred_qed, pred_sas, pred_map3k1, pred_pik3ca, pred_tp53  = bio_model(init_hidden_state, b_mutation_idx)

            ce_loss = F.cross_entropy(logits.permute(0, 2, 1), b_ids, reduction='sum') / args.batch_size

            logp_loss = F.mse_loss(pred_logp, b_logp, reduction='mean')
            qed_loss = F.mse_loss(pred_qed, b_qed,reduction='mean')
            sas_loss = F.mse_loss(pred_sas, b_sas,reduction='mean')

            map3k1_loss = F.mse_loss(pred_map3k1, b_map3k1,reduction='mean')
            pik3ca_loss = F.mse_loss(pred_pik3ca, b_pik3ca,reduction='mean')
            tp53_loss = F.mse_loss(pred_tp53, b_tp53,reduction='mean')

            loss =  args.beta *ce_loss + args.phi*logp_loss + args.gamma*qed_loss + args.delta*sas_loss + args.theta *map3k1_loss + args.theta *pik3ca_loss + args.theta *tp53_loss

            if num_batches % 100 == 0:
                print(f"ce loss :: {ce_loss :.3f}")
                print(f"logp loss :: {logp_loss :.3f}, qed loss :: {qed_loss :.3f}, sas loss :: {sas_loss :.3f}")
                print(f"map3k1 loss :: {map3k1_loss :.3f}, pik3ca loss :: {pik3ca_loss :.3f}, tp53 loss :: {tp53_loss :.3f}")


            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_ce_loss += args.beta*ce_loss.item()
            train_logp_loss += args.phi*logp_loss.item()
            train_qed_loss += args.gamma*qed_loss.item()
            train_sas_loss += args.delta*sas_loss.item()

            train_map3k1_loss += args.theta *map3k1_loss.item()
            train_pik3ca_loss += args.theta *pik3ca_loss.item()
            train_tp53_loss +=  args.theta *tp53_loss.item()

            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_ce_loss = train_ce_loss / num_batches
        train_logp_loss = train_logp_loss / (num_batches)
        train_qed_loss = train_qed_loss / (num_batches)
        train_sas_loss = train_sas_loss / (num_batches)

        train_map3k1_loss = train_map3k1_loss / num_batches
        train_pik3ca_loss = train_pik3ca_loss / num_batches
        train_tp53_loss = train_tp53_loss / num_batches

    
        dev_loss, dev_ce_loss, dev_logp_loss, dev_qed_loss, dev_sas_loss, dev_map3k1_loss, dev_pik3ca_loss, dev_tp53_loss = model_evaluate_bio(dev_dataloader, model,bio_model, device, args)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print('Saving model')
            if os.path.exists(args.filepath_mod):
                os.remove(args.filepath_mod)
            save_model(bio_model, optimizer, args, config, args.filepath_mod)

            if os.path.exists(args.filepath_mod_backbone):
                os.remove(args.filepath_mod_backbone)
            save_model(model, optimizer, args, config, args.filepath_mod_backbone)


        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}")

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        train_ce_losses.append(train_ce_loss)
        train_logp_losses.append(train_logp_loss)
        train_qed_losses.append(train_qed_loss)
        train_sas_losses.append(train_sas_loss)

        train_map3k1_losses.append(train_map3k1_loss)
        train_pik3ca_losses.append(train_pik3ca_loss)
        train_tp53_losses.append(train_tp53_loss)



        dev_ce_losses.append(dev_ce_loss)
        dev_logp_losses.append(dev_logp_loss)
        dev_qed_losses.append(dev_qed_loss)
        dev_sas_losses.append(dev_sas_loss)

        dev_map3k1_losses.append(dev_map3k1_loss)
        dev_pik3ca_losses.append(dev_pik3ca_loss)
        dev_tp53_losses.append(dev_tp53_loss)


    losses_df = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss' : dev_losses,
        'train_ce_loss':train_ce_losses,
        'train_logp': train_logp_losses,
        'train_qed': train_qed_losses,
        'train_sas_loss': train_sas_losses,
        'train_map3k1': train_map3k1_losses,
        'train_pik3ca': train_pik3ca_losses,
        'train_tp53' : train_tp53_losses,
        'val_ce_loss':dev_ce_losses,
        'val_logp': dev_logp_losses,
        'val_qed': dev_qed_losses,
        'val_sas_loss': dev_sas_losses,
        'val_map3k1' : dev_map3k1_losses,
        'val_pik3ca': dev_pik3ca_losses,
        'val_tp53': dev_tp53_losses
    })

    print('saving losses')

    

    losses_df.to_csv(args.filepath_losses)


