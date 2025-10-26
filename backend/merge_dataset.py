import pandas as pd
import os
from pathlib import Path

# ===================================
# PART 0: Rename column in dataset_1 
# ===================================
print("="*70)
print("STEP 0: RENAMING COLUMN IN DATASET 1")
print("="*70)

dataset_1_path = Path('data') / 'dataset_1_chunking_results.csv'
dataset_2_path = Path('data') / 'dataset_2_chunking_results.csv'

df_1 = pd.read_csv(dataset_1_path)

# List the name of columns before renaming
print(f"\nOriginal columns in dataset_1: {df_1.columns.tolist()}")

# Rename 'label' to 'is_vulnerable'
df_1 = df_1.rename(columns={'label': 'is_vulnerable'})
print(f"Rename Succeeded. New columns in dataset_1: {df_1.columns.tolist()}")
print(f"Dataset 1 shape: {df_1.shape}")


# ===================================
# PART 1: Rename column in dataset_2
# ===================================
print("="*70)
print("STEP 1: RENAMING COLUMN IN DATASET 2")
print("="*70)
df_2 = pd.read_csv(dataset_2_path)

# List the name of columns before renaming
print(f"\nOriginal columns in dataset_2: {df_2.columns.tolist()}")

# Rename 'PHP_Code' to 'code'
df_2 = df_2.rename(columns={'PHP_Code': 'code'})
print(f"Renamed 'PHP_Code' to 'code'. New columns in dataset_2: {df_2.columns.tolist()}")
print(f"Dataset 2 shape: {df_2.shape}")


# ===========================
# PART 2: Load other datasets
# ===========================
print("\n" + "="*70)
print("STEP 2: LOADING OTHER DATASETS")
print("="*70)

dataset_3_path = Path('data') / 'dataset_3_chunking_results.csv'
df_3 = pd.read_csv(dataset_3_path)

print(f"\nDataset 3 columns: {df_3.columns.tolist()}")
print(f"Dataset 3 shape: {df_3.shape}")


# ===========================
# PART 3: Merge all datasets
# ===========================
print("\n" + "="*70)
print("STEP 3: MERGING ALL DATASETS")
print("="*70)

# Keep only 'code' and 'is_vulnerable' columns from each dataset
df_1_filtered = df_1[['code', 'is_vulnerable']].copy()
df_2_filtered = df_2[['code', 'is_vulnerable']].copy()
df_3_filtered = df_3[['code', 'is_vulnerable']].copy()

# Merge all dataframes
merged_dataset = pd.concat([df_1_filtered, df_2_filtered, df_3_filtered], ignore_index=True)

print(f"\n✓ Successfully merged 3 datasets")
print(f"  Total rows: {len(merged_dataset):,}")
print(f"  Columns: {merged_dataset.columns.tolist()}")


# ===========================
# PART 4: Balance the dataset
# ===========================
print("\n" + "="*70)
print("STEP 4: BALANCING THE DATASET")
print("="*70)

# Show distribution before balancing
print(f"\nDistribution before balancing:")
print(merged_dataset['is_vulnerable'].value_counts().sort_index())
vuln_count = len(merged_dataset[merged_dataset['is_vulnerable'] == 1])
safe_count = len(merged_dataset[merged_dataset['is_vulnerable'] == 0])
print(f"  Vulnerable (1): {vuln_count:,}")
print(f"  Safe (0): {safe_count:,}")

# Separate vulnerable and safe samples
vulnerable_samples = merged_dataset[merged_dataset['is_vulnerable'] == 1]
safe_samples = merged_dataset[merged_dataset['is_vulnerable'] == 0]

# Balance the dataset
n_samples = min(len(vulnerable_samples), len(safe_samples))

print(f"\nBalancing to {n_samples:,} samples per class...")

balanced_dataset = pd.concat([
    vulnerable_samples.sample(n=n_samples, random_state=42),
    safe_samples.sample(n=n_samples, random_state=42)
], ignore_index=True)

# Shuffle the balanced dataset
balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nBalanced dataset created!")
print(f"  Total rows: {len(balanced_dataset):,}")
print(f"  Vulnerable (1): {len(balanced_dataset[balanced_dataset['is_vulnerable'] == 1]):,}")
print(f"  Safe (0): {len(balanced_dataset[balanced_dataset['is_vulnerable'] == 0]):,}")

# Show distribution after balancing
print(f"\n📊 Distribution after balancing:")
print(balanced_dataset['is_vulnerable'].value_counts().sort_index())



# ========================
# PART 5: Save the results
# ========================
print("\n" + "="*70)
print("STEP 5: SAVING RESULTS")
print("="*70)

# Save merged unbalanced dataset
merged_output_path = Path('data') / 'merged_all_datasets.csv'
merged_dataset.to_csv(merged_output_path, index=False)
print(f"\n✓ Merged dataset saved to: {merged_output_path}")

# Save balanced dataset
balanced_output_path = Path('data') / 'balanced_merged_dataset.csv'
balanced_dataset.to_csv(balanced_output_path, index=False)
print(f"✓ Balanced dataset saved to: {balanced_output_path}")

# =============
# FINAL SUMMARY
# =============
print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\n📊 Final Statistics:")
print(f"  Merged dataset: {len(merged_dataset):,} samples")
print(f"  Balanced dataset: {len(balanced_dataset):,} samples")
print(f"  Columns: {balanced_dataset.columns.tolist()}")
print(f"\n✅ All operations completed successfully!")