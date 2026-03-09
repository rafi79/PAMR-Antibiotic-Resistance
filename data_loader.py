"""
Data loading and preprocessing for AMR dataset.
"""
import pandas as pd
from typing import Dict, List, Tuple


def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, List[str], List[str], Dict]:
    """
    Load AMR dataset and identify variable genes.
    
    Args:
        data_path: Path to CSV dataset
        
    Returns:
        Tuple of (dataframe, gene_columns, class_columns, variable_genes_dict)
    """
    df = pd.read_csv(data_path)
    
    # Identify gene and class columns
    gene_cols = [c for c in df.columns if c.startswith("gene_")]
    class_cols = [c for c in df.columns if c.startswith("class_")]
    
    # Convert to numeric
    for col in gene_cols + class_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    
    # Handle numeric columns
    for col in ["Genome_Length_BP", "GC_Content_Percent", 
                "total_amr_genes", "total_resistance_classes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    
    # Identify variable genes (5-45 positive samples)
    variable_genes = {
        col: df[col].values
        for col in gene_cols 
        if 5 <= df[col].sum() <= 45
    }
    
    # Select primary target (closest to 25 positives)
    primary_target = min(
        variable_genes,
        key=lambda g: abs(variable_genes[g].sum() - 25)
    )
    
    print(f"\nVariable gene targets : {len(variable_genes)}")
    print(f"Primary target        : {primary_target}  "
          f"pos={variable_genes[primary_target].sum()}")
    print("\nAll targets:")
    for g, v in variable_genes.items():
        print(f"  {g.replace('gene_',''):<45}  "
              f"pos={v.sum()}  neg={50-v.sum()}")
    
    return df, gene_cols, class_cols, variable_genes


def get_gene_columns(df: pd.DataFrame) -> List[str]:
    """Extract gene column names from dataframe."""
    return [c for c in df.columns if c.startswith("gene_")]


def get_class_columns(df: pd.DataFrame) -> List[str]:
    """Extract class column names from dataframe."""
    return [c for c in df.columns if c.startswith("class_")]
