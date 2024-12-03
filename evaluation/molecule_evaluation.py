import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging to capture invalid SMILES strings
logging.basicConfig(filename='invalid_smiles.log', level=logging.WARNING,
                    format='%(asctime)s - %(message)s')


# Load SMILES data from CSV file
def load_smiles(file_path):
    df = pd.read_csv(file_path)
    return df['SMILES'].tolist()


# Convert SMILES to RDKit Mol objects and log any invalid SMILES
def smiles_to_mol(smiles_list):
    mols = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            mols.append(mol)
        else:
            logging.warning(f"Invalid SMILES: {sm}")
    return mols


# Calculate chemical properties (LogP, TPSA, MW, QED, SAS)
def calculate_properties(mol):
    try:
        logP = Descriptors.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        mw = Descriptors.MolWt(mol)
        qed = QED.qed(mol)
        sa_score = rdMolDescriptors.CalcSyntheticAccessibilityScore(mol)
        return logP, tpsa, mw, qed, sa_score
    except Exception as e:
        logging.warning(f"Error calculating properties for molecule: {e}")
        return None


# Check for Lipinski's Rule of Five compliance
def lipinski_filter(mol):
    mw = Descriptors.MolWt(mol)
    logP = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    return mw <= 500 and logP <= 5 and hbd <= 5 and hba <= 10


# Apply Lipinski filter and return compliant molecules and their count
def apply_lipinski_filter(mols):
    compliant_mols = [mol for mol in mols if lipinski_filter(mol)]
    return compliant_mols, len(compliant_mols)


# Generate Morgan fingerprints for similarity and novelty analysis
def get_fingerprints(mols):
    return [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]


# Compute Tanimoto similarity matrix
def calculate_similarity_matrix(fingerprints):
    n = len(fingerprints)
    tanimoto_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            tanimoto_matrix[i, j] = similarity
            tanimoto_matrix[j, i] = similarity
    return tanimoto_matrix


# Assess novelty of generated molecules compared to reference set (ZINC250K or other main molecule dataset)
def assess_novelty_against_reference(generated_fps, reference_fps):
    novelty_scores = []
    for gen_fp in generated_fps:
        max_similarity = max(DataStructs.TanimotoSimilarity(gen_fp, ref_fp) for ref_fp in reference_fps)
        novelty_scores.append(1 - max_similarity)  # Higher score indicates greater novelty
    return novelty_scores


# Determine scaffold novelty compared to reference molecules
def scaffold_novelty(generated_mols, reference_mols):
    generated_scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(mol=mol) for mol in generated_mols}
    reference_scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(mol=mol) for mol in reference_mols}
    novel_scaffolds = generated_scaffolds - reference_scaffolds
    return len(novel_scaffolds), novel_scaffolds


# Calculate diversity index from the Tanimoto distance matrix
def calculate_diversity_index(tanimoto_matrix):
    distance_matrix = 1 - tanimoto_matrix  # Convert similarity to distance
    diversity_index = np.mean(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
    return diversity_index


# Plot histograms of chemical property distributions
def plot_distributions(logP_list, tpsa_list, mw_list, qed_list, sa_score_list):
    properties = [logP_list, tpsa_list, mw_list, qed_list, sa_score_list]
    titles = ["LogP", "TPSA", "Molecular Weight", "QED", "SAS"]

    plt.figure(figsize=(15, 10))
    for i, prop in enumerate(properties):
        plt.subplot(2, 3, i + 1)
        plt.hist(prop, bins=30, alpha=0.7, color='blue')
        plt.title(titles[i])
        plt.xlabel(titles[i])
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# Main function for evaluation of generated molecules
def main():
    generated_file_path = 'generated_smiles.csv'  #Update
    reference_file_path = 'reference_smiles.csv'  #Update

    # Load and convert SMILES to RDKit Mol objects
    generated_smiles = load_smiles(generated_file_path)
    reference_smiles = load_smiles(reference_file_path)
    generated_mols = smiles_to_mol(generated_smiles)
    reference_mols = smiles_to_mol(reference_smiles)

    # Calculate properties for generated molecules
    properties = [calculate_properties(mol) for mol in generated_mols if calculate_properties(mol)]
    if not properties:
        print("No valid molecules to calculate properties.")
        return

    # Extract and analyze chemical properties
    logP_list, tpsa_list, mw_list, qed_list, sa_score_list = zip(*properties)

    # Apply Lipinski filter and get compliant molecule count
    compliant_mols, num_compliant = apply_lipinski_filter(generated_mols)
    print(f"Number of Lipinski-compliant molecules: {num_compliant}")

    # Generate fingerprints for both generated and reference molecules
    generated_fps = get_fingerprints(generated_mols)
    reference_fps = get_fingerprints(reference_mols)

    # Compute similarity matrix and diversity index
    tanimoto_matrix = calculate_similarity_matrix(generated_fps)
    diversity_index = calculate_diversity_index(tanimoto_matrix)

    # Assess novelty scores relative to the reference set
    novelty_scores = assess_novelty_against_reference(generated_fps, reference_fps)
    average_novelty = np.mean(novelty_scores)

    # Determine scaffold novelty and unique scaffold count
    num_novel_scaffolds, novel_scaffolds = scaffold_novelty(generated_mols, reference_mols)

    # Plot distributions of chemical properties
    plot_distributions(logP_list, tpsa_list, mw_list, qed_list, sa_score_list)

    # Print summary metrics
    print("Average LogP:", np.mean(logP_list))
    print("Average TPSA:", np.mean(tpsa_list))
    print("Average Molecular Weight:", np.mean(mw_list))
    print("Average QED:", np.mean(qed_list))
    print("Average SAS:", np.mean(sa_score_list))
    print("Average Novelty Score:", average_novelty)
    print("Number of Novel Scaffolds:", num_novel_scaffolds)
    print("Diversity Index:", diversity_index)


# Execute the script
if __name__ == "__main__":
    main()