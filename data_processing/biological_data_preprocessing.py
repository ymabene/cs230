import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path, sep="\t", skiprows=None):
    """
    Load dataset into a DataFrame.
    Parameters:
        file_path (str): Path to the data file.
        sep (str): Delimiter used in the file (default: tab).
        skiprows (int or list): Rows to skip - needed for additional column heads in clinical data files.
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        logging.info(f"Loading data from {file_path}...")
        data = pd.read_csv(file_path, sep=sep, skiprows=skiprows)
        logging.info(f"Data loaded. Shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        raise

def process_mutational_data(mutational_data, genes_of_interest):
    """
    Filter and reshape mutational data for selected genes.
    Parameters:
        mutational_data (pd.DataFrame): Raw mutation data.
        genes_of_interest (list): Target genes for filtering.
    Returns:
        pd.DataFrame: Pivoted mutation data with binary indicators.
    """
    try:
        logging.info("Processing mutational data...")
        filtered = mutational_data[mutational_data["Hugo_Symbol"].isin(genes_of_interest)].copy()
        filtered["Mutation"] = 1
        pivoted = filtered.pivot_table(index="Tumor_Sample_Barcode",
                                       columns="Hugo_Symbol",
                                       values="Mutation",
                                       aggfunc="first").fillna(0)
        pivoted.columns = [f"mutation_{col}" for col in pivoted.columns]
        logging.info(f"Processed mutational data. Shape: {pivoted.shape}")
        return pivoted.reset_index()
    except Exception as e:
        logging.error(f"Error processing mutational data: {e}")
        raise

def process_clinical_data(patient_data, sample_data):
    """
    Process clinical metadata and map sample to patient IDs.
    Parameters:
        patient_data (pd.DataFrame): Clinical patient data.
        sample_data (pd.DataFrame): Sample-to-patient mappings.
    Returns:
        tuple: Processed clinical data and sample mapping.
    """
    try:
        logging.info("Processing clinical data...")
        clinical = patient_data[["PATIENT_ID", "SUBTYPE", "CANCER_TYPE_ACRONYM"]].rename(
            columns={"SUBTYPE": "Subtype", "CANCER_TYPE_ACRONYM": "Cancer_Type"})
        mapping = sample_data[["SAMPLE_ID", "PATIENT_ID"]]
        logging.info(f"Processed clinical data. Shape: {clinical.shape}")
        return clinical, mapping
    except Exception as e:
        logging.error(f"Error processing clinical data: {e}")
        raise

def process_expression_data(expression_data, genes_of_interest):
    """
    Filter and reshape expression data for selected genes.
    Parameters:
        expression_data (pd.DataFrame): Gene expression data.
        genes_of_interest (list): Target genes for filtering.
    Returns:
        pd.DataFrame: Transposed expression data.
    """
    try:
        logging.info("Processing expression data...")
        filtered = expression_data[expression_data["Hugo_Symbol"].isin(genes_of_interest)].copy()
        transposed = filtered.set_index("Hugo_Symbol").transpose().rename_axis("SAMPLE_ID")
        transposed.columns = [f"expression_{col}" for col in transposed.columns]
        logging.info(f"Processed expression data. Shape: {transposed.shape}")
        return transposed.reset_index()
    except Exception as e:
        logging.error(f"Error processing expression data: {e}")
        raise

def merge_all_data(mutational_data, clinical_data, sample_mapping, expression_data):
    """
    Merge processed mutational, clinical, and expression data.
    Parameters:
        mutational_data (pd.DataFrame): Processed mutation data.
        clinical_data (pd.DataFrame): Clinical data.
        sample_mapping (pd.DataFrame): Sample-to-patient mappings.
        expression_data (pd.DataFrame): Processed expression data.
    Returns:
        pd.DataFrame: Unified dataset.
    """
    try:
        logging.info("Merging all data...")
        # Merge mutational data with sample-patient mapping
        merged_data = mutational_data.merge(
            sample_mapping, left_on="Tumor_Sample_Barcode", right_on="SAMPLE_ID", how="left"
        )
        # Merge with clinical data
        merged_data = merged_data.merge(
            clinical_data, on="PATIENT_ID", how="left"
        )
        # Merge with expression data
        final_data = merged_data.merge(
            expression_data, how="left", left_on="Tumor_Sample_Barcode", right_on="SAMPLE_ID"
        )

        # Drop redundant columns
        if "SAMPLE_ID_x" in final_data.columns and "SAMPLE_ID_y" in final_data.columns:
            final_data.drop(columns=["SAMPLE_ID_x"], inplace=True)
            final_data.rename(columns={"SAMPLE_ID_y": "SAMPLE_ID"}, inplace=True)
        elif "SAMPLE_ID" in final_data.columns:
            final_data.drop(columns=["SAMPLE_ID"], inplace=True)

        logging.info(f"Merged data. Shape: {final_data.shape}")
        return final_data
    except Exception as e:
        logging.error(f"Error during merging: {e}")
        raise

def add_pathway_activation(final_data, z_score_threshold=2.5):
    """
    Add pathway activation features based on mutations and expression data.
    Parameters:
        final_data (pd.DataFrame): Unified dataset.
        z_score_threshold (float): Z-score threshold for expression-based activation.
    Returns:
        pd.DataFrame: Dataset with added pathway features.
    """
    try:
        logging.info("Adding pathway activation features...")
        final_data["MAPK_Pathway_Activation"] = (
            (final_data["mutation_MAP3K1"] == 1) | (final_data["expression_MAP3K1"].abs() > z_score_threshold)
        ).astype(int)
        final_data["PI3K_Pathway_Activation"] = (
            (final_data["mutation_PIK3CA"] == 1) | (final_data["expression_PIK3CA"].abs() > z_score_threshold)
        ).astype(int)
        final_data["p53_Pathway_Activation"] = (
            (final_data["mutation_TP53"] == 1) | (final_data["expression_TP53"].abs() > z_score_threshold)
        ).astype(int)
        logging.info("Pathway activation features added.")
        return final_data
    except Exception as e:
        logging.error(f"Error adding pathway activation features: {e}")
        raise

def main():
    """
    Main execution pipeline.
    Loads, processes, merges data, and exports final results.
    """
    try:
        # Input file paths
        mutational_file = "data_mutations.txt"
        patient_file = "data_clinical_patient.txt"
        sample_file = "data_clinical_sample.txt"
        expression_file = "data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt"

        # Load data
        mutational_data = load_data(mutational_file)
        patient_data = load_data(patient_file, skiprows=4)
        sample_data = load_data(sample_file, skiprows=4)
        expression_data = load_data(expression_file)

        # Genes of interest
        genes_of_interest = ["PIK3CA", "TP53", "MAP3K1"]

        # Process data
        mutational = process_mutational_data(mutational_data, genes_of_interest)
        clinical, mapping = process_clinical_data(patient_data, sample_data)
        expression = process_expression_data(expression_data, genes_of_interest)

        # Merge data
        final_data = merge_all_data(mutational, clinical, mapping, expression)

        # Add pathway activations
        final_data = add_pathway_activation(final_data)

        # Reorder columns
        desired_columns = [
            "SAMPLE_ID", "PATIENT_ID", "Cancer_Type", "Subtype",
            "mutation_MAP3K1", "mutation_PIK3CA", "mutation_TP53",
            "expression_MAP3K1", "expression_PIK3CA", "expression_TP53",
            "MAPK_Pathway_Activation", "PI3K_Pathway_Activation", "p53_Pathway_Activation"
        ]
        final_data = final_data[desired_columns]

        # Export results
        cancer_type = "BRCA"  # Update for other datasets
        output_file = f"processed_{cancer_type}_data_with_pathways.csv"
        final_data.to_csv(output_file, index=False)
        logging.info(f"Final data exported to {output_file}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
