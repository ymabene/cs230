import torch
import torch.nn as nn
from torch.optim import Adam
import math
import torch.nn.functional as F


class QEDMLP_BIO(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(QEDMLP_BIO, self).__init__()
        self.layers = nn.Sequential(

            
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()

        )
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
    


class logpMLP_BIO(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(logpMLP_BIO, self).__init__()
        self.layers = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),

        )
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
    


class SASMLP_BIO(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(SASMLP_BIO, self).__init__()
        self.layers = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),

        )
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
    


class MAP3K1_BIO(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(MAP3K1_BIO, self).__init__()
        self.layers = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),

        )
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
      



class PIK3CA_BIO(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(PIK3CA_BIO, self).__init__()
        self.layers = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),

        )
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
      
        



class TP53_BIO(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(TP53_BIO, self).__init__()
        self.layers = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),

        )
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
      
        
        

class BIOMolformerModel(nn.Module):
    def __init__(self, config):
        super(BIOMolformerModel, self).__init__()

        self.mutation_vocab_size = config.mutation_vocab_size
        self.embeddings = nn.Embedding(self.mutation_vocab_size, config.dim)
        self.dim = config.dim
        self.SAS_head = SASMLP_BIO(hidden_dim=config.dim)
        self.logp_head = logpMLP_BIO(hidden_dim=config.dim)
        self.QED_head = QEDMLP_BIO(hidden_dim=config.dim)
        self.embedding_layer = nn.Linear(config.dim,config.dim)
       

    def forward(self, init_hidden_state, mutation_idx):

    
     
        hidden_state = self.embeddings(mutation_idx) # (bs, dim)
        hidden_state = self.embedding_layer(hidden_state) # (bs,dim)

      
        hidden_state = hidden_state + init_hidden_state.mean(dim = 1) # average over sequence length

    
        SAS = self.SAS_head(hidden_state)  # bs,dim -> bs
        QED = self.QED_head(hidden_state)
        logp = self.logp_head(hidden_state)

        map3k1 = self.logp_head(hidden_state)
        pik3ca = self.logp_head(hidden_state)
        tp53 = self.logp_head(hidden_state)


      
        return hidden_state, logp, QED, SAS, map3k1, pik3ca, tp53    


        