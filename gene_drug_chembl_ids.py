import pandas as pd
import requests
from io import StringIO
import re


def fetch_dataset(url: str) -> pd.DataFrame:
    """
    Fetches the dataset from the provided URL and returns it as a pandas DataFrame.
    :param url: URL of the dataset
    :return: pandas DataFrame containing the dataset
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print("Dataset fetched successfully.")
        return pd.read_csv(StringIO(response.text), sep='\t')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching dataset: {e}")
        raise


def extract_id(text: str, pattern: str) -> str:
    """
    Extracts the first match of a regex pattern from a string.
    :param text: String to extract data from
    :param pattern: Regex pattern to match
    :return: Extracted string or None if no match
    """
    if not isinstance(text, str) or pd.isna(text):
        return None
    match = re.search(pattern, text)
    return match.group(0) if match else None


def clean_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the ChEMBL and DrugBank ID columns by extracting relevant IDs.
    :param df: Original DataFrame
    :return: DataFrame with cleaned ID columns
    """
    chembl_pattern = r'CHEMBL\d+'
    drugbank_pattern = r'DB\d+'
    df['ChEMBL'] = df['ChEMBL'].apply(lambda x: extract_id(x, chembl_pattern))
    df['DrugBank ID'] = df['DrugBank ID'].apply(lambda x: extract_id(x, drugbank_pattern))
    return df


def extract_matching_genes(targets: str, target_genes: list) -> list:
    """
    Extracts genes from the 'Targets' column that match a given list of target genes.
    :param targets: String of target genes separated by semicolons
    :param target_genes: List of genes to match
    :return: List of matching genes
    """
    if pd.isna(targets):
        return []
    return [gene for gene in target_genes if gene in targets.split('; ')]


def expand_rows(df: pd.DataFrame, target_genes: list) -> pd.DataFrame:
    """
    Expands rows in the DataFrame to associate each target gene with its ChEMBL ID.
    :param df: Filtered DataFrame containing matched genes
    :param target_genes: List of genes to match
    :return: Expanded DataFrame with 'Gene' and 'ChEMBL' columns
    """
    expanded_rows = []
    for _, row in df.iterrows():
        matched_genes = extract_matching_genes(row['Targets'], target_genes)
        for gene in matched_genes:
            expanded_rows.append({'Gene': gene, 'ChEMBL': row['ChEMBL']})
    return pd.DataFrame(expanded_rows)


def group_chembl_ids_by_gene(expanded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the expanded rows by gene and aggregates ChEMBL IDs into a single row.
    :param expanded_df: DataFrame containing expanded rows
    :return: Grouped DataFrame with 'Gene' and aggregated 'ChEMBL' IDs
    """
    return expanded_df.groupby('Gene')['ChEMBL'].apply(lambda ids: ', '.join(ids)).reset_index()


def main():
    # Dataset URL
    url = 'https://sciencedata.anticancerfund.org/pages/cancerdrugsdb.txt'

    # List of target genes
    target_genes = ['PIK3CA', 'TP53', 'MAP2K1']

    # Step 1: Fetch and load dataset
    df = fetch_dataset(url)

    # Step 2: Clean the ChEMBL and DrugBank ID columns
    df = clean_ids(df)

    # Step 3: Filter rows where 'Targets' contains any of the target genes
    filtered_df = df[df['Targets'].apply(lambda x: any(gene in str(x).split('; ') for gene in target_genes) if pd.notna(x) else False)]

    # Step 4: Expand rows to associate genes with ChEMBL IDs
    expanded_df = expand_rows(filtered_df, target_genes)

    # Step 5: Group by gene and aggregate ChEMBL IDs
    grouped_gene_chembl = group_chembl_ids_by_gene(expanded_df)

    # Step 6: Save results to a CSV file
    output_file = 'gene_chembl_ids.csv'
    grouped_gene_chembl.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(grouped_gene_chembl)


if __name__ == "__main__":
    main()
