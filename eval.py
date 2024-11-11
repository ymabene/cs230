import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score



from tqdm import tqdm

from pretraining_scripts import *





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the molformer parameters are frozen; finetune: molformer parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=767)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=80)
    parser.add_argument("--gamma", help = "scaling factor for QED loss",type=float, default=12)
    parser.add_argument("--delta", help = "scaling factor for SAS loss",type=float, default=3)
    parser.add_argument("--phi", help = "scaling factor for logp loss",type=float, default=3)
    parser.add_argument("--beta", help = "scaling factor for cross entropy loss",type=float, default=1)
    parser.add_argument("--filepath_inference", help = "path to save model outputs",type=str, default='outputs/equal_scaling/zinc_pretrained_inference_111.csv')
    parser.add_argument("--saved_model_filepath", help = "path to saved model",type=str, default='outputs\zinc_pretrained_111.pt')
 



    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
   
    print('Evaluating Molformer on Zinc...')
    config = SimpleNamespace(
        filepath='molformer-zinc.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train_path='data/zinc-250k-train.csv',
        dev_path='data/zinc-250k-val.csv',
        test_path='data/zinc-250k-test.csv',
        option=args.option,
        num_heads = args.num_heads,
        num_layers = args.num_layers,
        dim = args.dim,
        max_seq_len = args.max_seq_len,
        vocab_size = args.vocab_size

    )


   
    evaluate(args,config)

 
   