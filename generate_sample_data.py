import pandas as pd
import os

def create_sample_data(input_file, output_file, num_rows=10, sep="\t"):
    """
    Create a smaller sample dataset from a larger file.

    Parameters:
        input_file (str): Path to the input file.
        output_file (str): Path to save the sampled data.
        num_rows (int): Number of rows to sample.
        sep (str): Delimiter for the input file (default: tab-separated).
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")

        # Load the full dataset
        print(f"Loading data from {input_file}...")
        data = pd.read_csv(input_file, sep=sep)
        print(f"Original data shape: {data.shape}")

        # Sample the data
        sampled_data = data.head(num_rows)
        print(f"Sampled data shape: {sampled_data.shape}")

        # Save the sampled dataset
        sampled_data.to_csv(output_file, sep=sep, index=False)
        print(f"Sampled data saved to {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    """
    Main function to create smaller sample datasets from larger files.
    """
    # File paths for full datasets
    mutational_file = "data_mutations.txt"
    clinical_patient_file = "data_clinical_patient.txt"
    clinical_sample_file = "data_clinical_sample.txt"
    expression_file = "data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt"

    # Output paths for sampled datasets
    sampled_mutational_file = "sample_mutational_data.txt"
    sampled_clinical_patient_file = "sample_clinical_patient_data.txt"
    sampled_clinical_sample_file = "sample_clinical_sample_data.txt"
    sampled_expression_file = "sample_expression_data.txt"

    # Number of rows to include in the sample
    num_rows = 10

    # Generate sample datasets
    create_sample_data(mutational_file, sampled_mutational_file, num_rows)
    create_sample_data(clinical_patient_file, sampled_clinical_patient_file, num_rows)
    create_sample_data(clinical_sample_file, sampled_clinical_sample_file, num_rows)
    create_sample_data(expression_file, sampled_expression_file, num_rows)

if __name__ == "__main__":
    main()
