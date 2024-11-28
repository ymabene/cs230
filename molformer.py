import torch
import torch.nn as nn
from torch.optim import Adam
import math
import torch.nn.functional as F
import random


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        self.head_dim =config.dim // config.num_heads
        self.num_heads = config.num_heads
        self.max_seq_len = config.max_seq_len

        self.wq = nn.Linear(config.dim, config.num_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(config.dim, config.num_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(config.dim, config.num_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, config.dim, bias = False)


    
    def forward(self, x, device):

        batch_size, seq_len, _ = x.shape

        xq = self.wq(x) # (b,s,d) * (d, num_heads * head_d) = (b,s,num_heads*head_d)
        xk = self.wk(x)
        xv = self.wv(x)

        # Split last dim

        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Reshape

        xq = xq.transpose(1,2) # (b, num_heads, s, head_dim)
        xk =xk.transpose(1,2)
        xv = xv.transpose(1,2)

        # Find attention scores

        scores = torch.matmul(xq, xk.transpose(-1,-2)) / math.sqrt(self.head_dim)
        norm_scores = F.softmax(scores, dim = -1)
        
        weighted_scores = torch.matmul(norm_scores,xv)
        weighted_scores = weighted_scores.transpose(1,2)

        weighted_scores = weighted_scores.reshape(weighted_scores.size()[0], weighted_scores.size()[1],
                                                  self.head_dim * self.num_heads)
        
        output = self.wo(weighted_scores) # (bs, seq_len, dim)

        return output
    

class FeedForward(nn.Module):
    def __init__(self,config):
        super(FeedForward, self).__init__()

        hidden_dim = config.dim

        hidden_dim *=4
        hidden_dim = int(2 * hidden_dim/3)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias = False)
        self.v = nn.Linear(config.dim, hidden_dim, bias = False)


    def forward(self, x):

        # x: (bs, seq_len, dim)

        swish = F.silu(self.w1(x))
        product = (swish * self.v(x))
        output = self.w2(product)

        return output
        


class RMSNorm(nn.Module):
    # Pytorch implementation
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale parameter
        self.bias = nn.Parameter(torch.zeros(dim))   # Learnable bias parameter

    def forward(self, x):
        # Calculate the RMS along the last dimension (feature dimension)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize the input tensor
        normalized_x = x / rms
        # Apply scale and bias
        return normalized_x * self.weight + self.bias
    


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.num_heads = config.num_heads
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.dim)
        self.feed_forward_layer = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, x, mask, device):


        h = self.attention(self.attention_norm(x), device) + x

        output = h + self.feed_forward_layer(self.ffn_norm(h))

        return output.to(device)



class PositionalEncoding(nn.Module):

    # Pytorch implementation

    def __init__(self, d_model,max_len):
        super(PositionalEncoding, self).__init__()
      
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :] # (bs, seq_len, dim)
        return x



class QEDMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(QEDMLP, self).__init__()
        self.layers = nn.Sequential(

            
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()

        )
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
    


class logpMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(logpMLP, self).__init__()
        self.layers = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),

        )
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
    


class SASMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(SASMLP, self).__init__()
        self.layers = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),

        )
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
        
        

class MolformerModel(nn.Module):
    def __init__(self, config):
        super(MolformerModel, self).__init__()

        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.norm = RMSNorm(config.dim)
        self.output_layer = nn.Linear(config.dim, self.vocab_size, bias=False)
        self.embeddings = nn.Embedding(self.vocab_size, config.dim)
        self.head_dim = config.dim // config.num_heads
        self.max_seq_len = config.max_seq_len
        self.dim = config.dim
        self.pos_encoding = PositionalEncoding(config.dim, self.max_seq_len)
        self.SAS_head = SASMLP(hidden_dim=128)
        self.logp_head = logpMLP(hidden_dim=128)
        self.QED_head = QEDMLP(hidden_dim=128)
        self.mask = config.mask


        self.layers = torch.nn.ModuleList()

        for layers in range(config.num_layers):
            self.layers.append(Decoder(config))

        
        


    ## Padding masks
    def make_attention_masks(self, batch_input, device, pad_indx):

        batch_input_len = batch_input.shape[1]
        batch_size = batch_input.shape[0]

        batch_input_pad = (batch_input == pad_indx).float()

        batch_input_pad = batch_input_pad.masked_fill(batch_input_pad ==1, float('-inf')) # (bs, b_input_len)
        batch_input_pad = batch_input_pad.unsqueeze(1).unsqueeze(1).expand(-1,self.num_heads,-1,-1).to(device)
        # (bs, num_heads, 1, batch_input_len)

        batch_input_pad = batch_input_pad.to(device)

      

        return batch_input_pad
    




    def forward(self, input_id, pad_indx):

        device = input_id.device

       
        bs, seq_len = input_id.shape

        mask = self.make_attention_masks(input_id, device, pad_indx)

        hidden_state = self.embeddings(input_id) # (bs, seq_len, dim)

        if self.mask == True:
            random_index = random.randint(0,seq_len-1)
            hidden_state[:,random_index,:] = self.embeddings.weight[1] # randomly mask character

    
        # Add pos embeddings 
        hidden_state = self.pos_encoding(hidden_state)


        for layer in self.layers:
            hidden_state = layer(hidden_state, mask, device)  


        hidden_state = self.norm(hidden_state) # bs, seq_len, dim

        
        output = self.output_layer(hidden_state).float() # bs, seq_len, vocab size

        SAS = self.SAS_head(hidden_state.mean(dim=1))  # bs,vocab_dim -> bs
        QED = self.QED_head(hidden_state.mean(dim=1))
        logp = self.logp_head(hidden_state.mean(dim=1))

    
        return output, logp, QED, SAS, hidden_state   


