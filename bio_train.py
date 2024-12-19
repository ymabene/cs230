import argparse
from types import SimpleNamespace

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from finetune_script import *

# This script trains the BioMolformer Model





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
    parser.add_argument("--gamma", help = "scaling factor for QED loss",type=float, default=4) 
    parser.add_argument("--delta", help = "scaling factor for SAS loss",type=float, default=1) 
    parser.add_argument("--phi", help = "scaling factor for logp loss",type=float, default=1) 
    parser.add_argument("--theta", help = "scaling factor for expression losses",type=float, default=3)
    parser.add_argument("--beta", help = "scaling factor for cross entropy loss",type=float, default=0.3)
    parser.add_argument("--filepath_mod", help = "path to save model checkpoints",type=str, default='outputs/masking/bio_model_finetuned.pt')
    parser.add_argument("--filepath_mod_backbone", help = "path to save backbone model checkpoints",type=str, default='outputs/masking/bio_model_backbone_finetuned.pt')
    parser.add_argument("--filepath_losses", help = "path to save model losses",type=str, default='outputs/masking/bio_model_finetuned_losses.csv')
    parser.add_argument("--saved_model_filepath", help = "path to saved backbone model",type=str, default='outputs/default_scaling/zinc_pretrained.pt')
    parser.add_argument("--mutation_vocab_size", help = "vocab size for gene mutations",type=int, default=7)
    parser.add_argument("--mask", help = "whether or not to conduct masking of input during training",type=bool, default=False)
    
 


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
   
    print('Training BIOMolformer on Zinc for epochs:', args.epochs)
    config = SimpleNamespace(
        filepath= args.filepath_mod,
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train_path='data/bio-chem-train.csv',
        dev_path='data/bio-chem-val.csv',
        test_path='data/bio-chem-test.csv',
        option=args.option,
        num_heads = args.num_heads,
        num_layers = args.num_layers,
        dim = args.dim,
        max_seq_len = args.max_seq_len,
        vocab_size = args.vocab_size,
        mutation_vocab_size = args.mutation_vocab_size,
        mask = args.mask
    )



    train(args,config)

 
   