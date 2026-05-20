"""
Dataset Verification Script for Kelompok 08.
This script checks the integrity, shapes, data types, file sizes, and class balance
of the processed dataset splits (.npy arrays) and the scaler (.pkl).

Usage:
    python verify_dataset.py --splits-dir <path_to_splits>
"""

import argparse
import os
from pathlib import Path
import numpy as np
import joblib

def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"

def verify_splits(splits_dir):
    splits_dir = Path(splits_dir)
    print("=" * 60)
    print(f"VERIFYING DATASET SPLITS IN: {splits_dir.resolve()}")
    print("=" * 60)

    if not splits_dir.exists():
        print(f"ERROR: Splits directory '{splits_dir}' does not exist.")
        return

    # List of files we expect
    flat_files = ['X_train.npy', 'X_val.npy', 'X_test.npy', 'y_train.npy', 'y_val.npy', 'y_test.npy']
    seq_files = ['X_train_seq.npy', 'X_val_seq.npy', 'X_test_seq.npy', 'y_train_seq.npy', 'y_val_seq.npy', 'y_test_seq.npy']
    scaler_file = 'scaler.pkl'

    all_present = True

    print("\n--- 1. File Presence & Sizes ---")
    for f in flat_files + seq_files + [scaler_file]:
        fpath = splits_dir / f
        if not fpath.exists():
            print(f"  [MISSING] {f}")
            all_present = False
        else:
            sz = fpath.stat().st_size
            print(f"  [FOUND]   {f:<16} | Size: {format_size(sz)}")

    if not all_present:
        print("\nWARNING: Some expected splits/scaler files are missing!")

    print("\n--- 2. Shape, Dtype & Class Balance Verification ---")
    # Load and check flat splits
    try:
        X_train = np.load(splits_dir / 'X_train.npy')
        y_train = np.load(splits_dir / 'y_train.npy')
        X_val = np.load(splits_dir / 'X_val.npy')
        y_val = np.load(splits_dir / 'y_val.npy')
        X_test = np.load(splits_dir / 'X_test.npy')
        y_test = np.load(splits_dir / 'y_test.npy')

        print("\n[Flat splits (for Random Forest)]:")
        for name, X, y in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
            n_normal = int((y == 0).sum())
            n_ddos = int((y == 1).sum())
            total = len(y)
            normal_pct = (n_normal / total) * 100
            ddos_pct = (n_ddos / total) * 100
            print(f"  {name:<5} | X: {str(X.shape):<15} | y: {str(y.shape):<10} | "
                  f"Normal: {n_normal:,} ({normal_pct:.2f}%) | DDoS: {n_ddos:,} ({ddos_pct:.2f}%)")

    except Exception as e:
        print(f"  Error checking flat splits: {e}")

    # Load and check sequence splits
    try:
        X_train_seq = np.load(splits_dir / 'X_train_seq.npy')
        y_train_seq = np.load(splits_dir / 'y_train_seq.npy')
        X_val_seq = np.load(splits_dir / 'X_val_seq.npy')
        y_val_seq = np.load(splits_dir / 'y_val_seq.npy')
        X_test_seq = np.load(splits_dir / 'X_test_seq.npy')
        y_test_seq = np.load(splits_dir / 'y_test_seq.npy')

        print("\n[Sequence splits (for LSTM/GRU/CNN/Transformer)]:")
        for name, X_seq, y_seq in [('Train', X_train_seq, y_train_seq), ('Val', X_val_seq, y_val_seq), ('Test', X_test_seq, y_test_seq)]:
            n_normal = int((y_seq == 0).sum())
            n_ddos = int((y_seq == 1).sum())
            total = len(y_seq)
            normal_pct = (n_normal / total) * 100
            ddos_pct = (n_ddos / total) * 100
            print(f"  {name:<5} | X_seq: {str(X_seq.shape):<18} | y_seq: {str(y_seq.shape):<10} | "
                  f"Normal: {n_normal:,} ({normal_pct:.2f}%) | DDoS: {n_ddos:,} ({ddos_pct:.2f}%)")

    except Exception as e:
        print(f"  Error checking sequence splits: {e}")

    # Load and check scaler statistics
    try:
        scaler = joblib.load(splits_dir / 'scaler.pkl')
        print("\n--- 3. Scaler Properties ---")
        print(f"  Scaler Type:    {type(scaler).__name__}")
        print(f"  Scaled Features: {scaler.n_features_in_}")
        print(f"  Mean (first 5):  {scaler.mean_[:5].round(4)}")
        print(f"  Scale (first 5): {scaler.scale_[:5].round(4)}")
        
        # Verify flat X_train scaling
        if 'X_train' in locals():
            print(f"  X_train actual mean (should be ~0): {X_train.mean(axis=0)[:5].round(4)}")
            print(f"  X_train actual std  (should be ~1): {X_train.std(axis=0)[:5].round(4)}")

    except Exception as e:
        print(f"  Error checking scaler: {e}")
        
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits-dir', default='data/processed/splits', 
                        help='Path to splits folder (e.g. "G:/My Drive/tugas-akhir-ai/splits")')
    args = parser.parse_args()
    verify_splits(args.splits_dir)

