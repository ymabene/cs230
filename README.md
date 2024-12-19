**Repository for the Molformer Model**

This is the repository for the Molformer models. Molformer is a transformer based model trained on chemical and biological data to generate representations of small molecules.

Molformer model is within the folder called **model**

Molformer model using chemical property data is found in **molformer.py**

Molformer model using chemical property data and biological data is found in **biomolformer.py**

Training and inference scripts are in the following files:

**train.py**

**eval.py**

**bio_train.py**

**bio_eval.py**

**pretraining_scripts.py**

**finetune_script.py**

Data processing scripts are in the folder called **data_processing**


Evaluation scripts are found in the folder called **evaluation**

Data is found in the folder called **data**


To train and run inference on ChemMolformer model using Zinc-250k dataset:

```bash
python train.py

python eval.py
```

To train and run inference on BioMolformer model using integrated dataset:

```bash
python bio_train.py

python bio_eval.py
```

