"""Quick verification script for Phase 1 dataset."""
import os
import pandas as pd

DATA_DIR = "data"

print("=" * 60)
print("  Phase 1 Verification")
print("=" * 60)

# File sizes
print("\n--- FILE SIZES ---")
for fname in ["disputes.csv", "train.csv", "val.csv", "test.csv"]:
    fpath = os.path.join(DATA_DIR, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname}: {size:,} bytes")

# Full dataset schema
df = pd.read_csv(os.path.join(DATA_DIR, "disputes.csv"))
print("\n--- SCHEMA (20 columns) ---")
print(df.dtypes.to_string())

print("\n--- FIRST 3 RECORDS ---")
print(df.head(3).T.to_string())

print("\n--- CLASS BALANCE ---")
print(df["dispute_outcome"].value_counts().to_string())

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) == 0:
    print("  None (all columns complete)")
else:
    print(missing_cols.to_string())

print("\n--- CORRELATION WITH TARGET (top 10 by absolute value) ---")
numeric = df.select_dtypes(include="number")
corr = numeric.corr()["dispute_outcome"].drop("dispute_outcome").abs().sort_values(ascending=False)
print(corr.head(10).to_string())

# Verify temporal ordering of splits
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

train_max = train["dispute_date"].max()
val_min = val["dispute_date"].min()
val_max = val["dispute_date"].max()
test_min = test["dispute_date"].min()

print("\n--- TEMPORAL SPLIT INTEGRITY ---")
print(f"  Train latest:  {train_max}")
print(f"  Val earliest:  {val_min}")
print(f"  Val latest:    {val_max}")
print(f"  Test earliest: {test_min}")
print(f"  Train < Val:   {train_max <= val_min}")
print(f"  Val < Test:    {val_max <= test_min}")

print("\n[OK] Phase 1 verification complete.")
