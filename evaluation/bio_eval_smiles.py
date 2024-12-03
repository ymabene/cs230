import argparse
from types import SimpleNamespace

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from finetune_script import *

# Using CS224N final project config as reference





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
    parser.add_argument("--filepath_mod", help = "path to save model checkpoints",type=str, default='outputs/bio_1e-5/bio_model_finetuned.pt')
    parser.add_argument("--filepath_losses", help = "path to save model losses",type=str, default='outputs/bio_1e-5/bio_model_finetuned_losses.csv')
    parser.add_argument("--saved_model_filepath", help = "path to saved backbone model",type=str, default='outputs/bio_1e-5/bio_model_backbone_finetuned.pt')
    parser.add_argument("--filepath_inference", help = "path to save model outputs",type=str, default='outputs/bio_1e-5/bio_inference.csv')
    parser.add_argument("--mutation_vocab_size", help = "vocab size for gene mutations",type=int, default=7)
    parser.add_argument("--saved_bio_model_filepath", help = "path to save bio model checkpoints",type=str, default='outputs/bio_1e-5/bio_model_finetuned.pt')
    parser.add_argument("--mask", help = "whether or not to conduct masking of input during training",type=bool, default=False)
    

    args = parser.parse_args()
    return args

def evaluate_smiles(args, config):

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

    tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')

    smiles_list = []


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

        #hidden_state, pred_logp,  pred_qed, pred_sas, pred_map3k1, pred_pik3ca, pred_tp53  = bio_model(init_hidden_state, b_mutation_idx)

        probabilities = F.softmax(logits, dim=-1)


        token_ids = torch.argmax(probabilities, dim = -1)


        smiles = [tokenizer.decode(mol, skip_special_tokens=True) for mol in token_ids]

      

        smiles_list.extend(smiles)

    smiles_df = pd.DataFrame({'SMILES': smiles_list})
    smiles_df.to_csv('outputs/smiles_list.csv', index=False)



if __name__ == "__main__":
    args = get_args()
   
    print('Evaluating BIOMolformer on Zinc for epochs:', args.epochs)
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



    evaluate_smiles(args,config)

 
   