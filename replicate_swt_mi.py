#!/usr/bin/env python3
"""
Fixed: Sharma & Sunkaria (2018) - SWT + features + KNN/SVM on PTB-DB.
"""

import math
import os
from collections import Counter

import numpy as np
import pandas as pd
import pywt
import wfdb
from scipy.signal import medfilt, savgol_filter
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

# -------- PARAMETERS --------
DATA_DIR = "./ptbdb"
LEADS = ["ii", "iii", "avf"]  # lowercase for consistent matching
TARGET_FS = 250
SEG_S = 3.072
SEG_LEN = int(TARGET_FS * SEG_S)  # 768
WAVELET = "db5"
SWT_LEVELS = 6
SAMP_ENT_M = 2
SAMP_ENT_R_FACTOR = 0.2
N_GAIN_BINS = 10
TOP_K_FEATURES = 10
EPS = 1e-12


def get_subject_id(record_name):
    return record_name.split('_')[0].split('/')[0]


def downsample_signal(sig, orig_fs, target_fs=TARGET_FS):
    if orig_fs == target_fs:
        return sig
    factor = int(round(orig_fs / target_fs))
    return sig[::factor]


def preprocess_signal(sig, fs=TARGET_FS):
    w1 = int(0.2 * fs) | 1
    w2 = int(0.6 * fs) | 1
    baseline = medfilt(sig, kernel_size=w1)
    baseline = medfilt(baseline, kernel_size=w2)
    z = sig - baseline
    win = int(0.05 * fs) | 1
    if win < 5: win = 5
    z = savgol_filter(z, win, polyorder=3)
    return z


def sample_entropy_fast(x, m=SAMP_ENT_M, r_factor=SAMP_ENT_R_FACTOR):
    """Optimized sample entropy - trades some accuracy for massive speed gain"""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    # For very long signals, downsample for SampEn calculation only
    if N > 200:
        step = N // 200
        x = x[::step]
        N = len(x)

    r = r_factor * np.std(x)
    if r == 0 or N < m + 2:
        return 0.0

    # Simplified counting using numpy operations
    B_count = 0
    A_count = 0

    for i in range(N - m):
        # Get template of length m
        template_m = x[i:i + m]

        # Compare with all subsequent templates (upper triangle only)
        for j in range(i + 1, N - m + 1):
            dist_m = np.max(np.abs(template_m - x[j:j + m]))
            if dist_m <= r:
                B_count += 1

                # Check m+1 length
                if i < N - m - 1 and j < N - m:
                    dist_m1 = np.max(np.abs(x[i:i + m + 1] - x[j:j + m + 1]))
                    if dist_m1 <= r:
                        A_count += 1

    if B_count == 0:
        return 0.0
    return -np.log((A_count + EPS) / (B_count + EPS))


def segment_signal(signal, seg_len=SEG_LEN):
    n = len(signal)
    segments = []
    for start in range(0, n - seg_len + 1, seg_len):
        segments.append(signal[start:start + seg_len])
    return segments


def load_record_signal(rec, data_dir, leads):
    """Load signal with case-insensitive lead matching"""
    rec_path = os.path.join(data_dir, rec).replace("\\", "/")

    try:
        sig, fields = wfdb.rdsamp(rec_path)
    except Exception as e:
        raise ValueError(f"Cannot read record {rec}: {e}")

    # PTB database uses various lead name formats
    sig_names = [s.strip().lower() for s in fields['sig_name']]
    fs = int(fields['fs'])

    lead_sigs = []
    for ld in leads:
        ld_norm = ld.strip().lower()
        found = False

        # Try exact match first
        if ld_norm in sig_names:
            idx = sig_names.index(ld_norm)
            lead_sigs.append(sig[:, idx])
            found = True
        else:
            # Try common variations (avf vs AVF, etc.)
            for variant in [ld_norm, ld_norm.upper(), ld_norm.capitalize()]:
                if variant.lower() in sig_names:
                    idx = sig_names.index(variant.lower())
                    lead_sigs.append(sig[:, idx])
                    found = True
                    break

        if not found:
            lead_sigs.append(None)

    return lead_sigs, fs


def extract_features_from_coeffs(coeff, fs=TARGET_FS):
    """Extract 4 features from wavelet coefficients"""
    total_energy = np.sum(coeff ** 2)
    return {
        'SEN': sample_entropy_fast(coeff),
        'NSE': total_energy,  # normalized later
        'LEE': np.sum(np.log2(coeff ** 2 + EPS)),
        'MDS': np.median(np.abs(np.diff(coeff))) * fs
    }


def build_dataset(records, data_dir=DATA_DIR, leads=None):
    if leads is None:
        leads = LEADS

    # Load labels
    labels_file = os.path.join(data_dir, "PTB_LABELS.csv")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"PTB_LABELS.csv not found")

    dfmap = pd.read_csv(labels_file, dtype=str)
    dfmap['record'] = dfmap['record'].str.strip()
    dfmap['label'] = dfmap['label'].str.strip().str.upper()

    rows = []
    labels_out = []
    subjects_out = []

    records_processed = 0
    records_skipped = 0
    segments_extracted = 0

    pbar = tqdm(records, desc="Processing records")
    for rec in pbar:
        # Match label
        m = dfmap[dfmap['record'] == rec]
        if m.empty:
            rec_basename = os.path.basename(rec)
            m = dfmap[dfmap['record'].apply(lambda x: os.path.basename(str(x)) == rec_basename)]
        if m.empty:
            records_skipped += 1
            continue

        label = m.iloc[0]['label']
        if label not in ("HC", "IMI"):
            records_skipped += 1
            continue

        subj = get_subject_id(rec)

        # Load signals
        try:
            lead_sigs, orig_fs = load_record_signal(rec, data_dir, leads)
        except Exception as e:
            records_skipped += 1
            pbar.set_postfix({'processed': records_processed, 'skipped': records_skipped,
                              'segments': segments_extracted})
            continue

        # Check all leads present
        if any(s is None for s in lead_sigs):
            records_skipped += 1
            pbar.set_postfix({'processed': records_processed, 'skipped': records_skipped,
                              'segments': segments_extracted})
            continue

        # Process each lead
        per_lead_segments = []
        for sig in lead_sigs:
            ds = downsample_signal(sig, orig_fs, TARGET_FS)
            if len(ds) < SEG_LEN:
                per_lead_segments = []
                break
            z = preprocess_signal(ds, TARGET_FS)
            segs = segment_signal(z, SEG_LEN)
            if not segs:
                per_lead_segments = []
                break
            per_lead_segments.append(segs)

        if not per_lead_segments:
            records_skipped += 1
            pbar.set_postfix({'processed': records_processed, 'skipped': records_skipped,
                              'segments': segments_extracted})
            continue

        # Process segments
        nseg = min(len(s) for s in per_lead_segments)
        for si in range(nseg):
            features = {}

            for li, lead in enumerate(leads):
                seg = per_lead_segments[li][si]

                # SWT decomposition for seg and seg^2
                for power_label, seg_power in [("", seg), ("_z2", seg ** 2)]:
                    try:
                        coeffs = pywt.swt(seg_power, WAVELET, level=SWT_LEVELS)
                    except:
                        continue

                    # Calculate total energy
                    total_e = sum(np.sum(d ** 2) for a, d in coeffs)
                    total_e += np.sum(coeffs[-1][0] ** 2)
                    if total_e == 0:
                        total_e = 1.0

                    # Extract features from detail coefficients
                    for lvl, (a, d) in enumerate(coeffs, start=1):
                        feats = extract_features_from_coeffs(d, TARGET_FS)
                        for fname, fval in feats.items():
                            key = f"{lead}_D{lvl}_{fname}{power_label}"
                            if fname == 'NSE':
                                features[key] = fval / total_e
                            else:
                                features[key] = fval

                    # Approximation coefficients
                    a_final = coeffs[-1][0]
                    feats = extract_features_from_coeffs(a_final, TARGET_FS)
                    for fname, fval in feats.items():
                        key = f"{lead}_A{SWT_LEVELS}_{fname}{power_label}"
                        if fname == 'NSE':
                            features[key] = fval / total_e
                        else:
                            features[key] = fval

            if features:
                rows.append(features)
                labels_out.append(1 if label == "IMI" else 0)
                subjects_out.append(subj)
                segments_extracted += 1

        records_processed += 1
        pbar.set_postfix({'processed': records_processed, 'skipped': records_skipped,
                          'segments': segments_extracted})

    pbar.close()
    print(f"\nProcessing complete:")
    print(f"  Records processed: {records_processed}")
    print(f"  Records skipped: {records_skipped}")
    print(f"  Total segments extracted: {segments_extracted}")

    X = pd.DataFrame(rows)
    y = np.array(labels_out, dtype=int)
    subjects = np.array(subjects_out, dtype=object)

    return X, y, subjects


def entropy_of_counts(counts):
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs + EPS))


def gain_ratio_feature(X_col, y, n_bins=N_GAIN_BINS):
    df = pd.DataFrame({"x": X_col, "y": y})
    try:
        df['bin'] = pd.qcut(df['x'], n_bins, duplicates='drop')
    except:
        df['bin'] = pd.cut(df['x'], n_bins)

    classes, counts = np.unique(y, return_counts=True)
    InfoT = entropy_of_counts(counts)

    N = len(df)
    weighted_info = 0.0
    split_info = 0.0

    for b in df['bin'].unique():
        if pd.isna(b):
            continue
        subset = df[df['bin'] == b]
        if subset.empty:
            continue
        _, ccounts = np.unique(subset['y'], return_counts=True)
        p = len(subset) / N
        weighted_info += p * entropy_of_counts(ccounts)
        split_info -= p * math.log2(p + EPS)

    gain = InfoT - weighted_info
    if split_info == 0:
        return 0.0
    return gain / split_info


def select_top_k_features(X_df, y, k=TOP_K_FEATURES):
    scores = {}
    for col in tqdm(X_df.columns, desc="Calculating gain ratios"):
        scores[col] = gain_ratio_feature(X_df[col].values, y)
    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    topk = [f for f, s in sorted_feats[:k]]
    return topk, scores


def evaluate_class_oriented(X, y, topk_features):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {'auc': [], 'acc': [], 'sen': [], 'spec': [], 'prec': []}

    for train_idx, test_idx in skf.split(X, y):
        Xtr = X.iloc[train_idx][topk_features].values
        Xte = X.iloc[test_idx][topk_features].values
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(Xtr, ytr)
        ypred = knn.predict(Xte)
        yproba = knn.predict_proba(Xte)[:, 1]

        metrics['auc'].append(roc_auc_score(yte, yproba))
        metrics['acc'].append(accuracy_score(yte, ypred))
        metrics['sen'].append(recall_score(yte, ypred))
        metrics['prec'].append(precision_score(yte, ypred))

        tn = np.sum((yte == 0) & (ypred == 0))
        fp = np.sum((yte == 0) & (ypred == 1))
        metrics['spec'].append(tn / (tn + fp + EPS))

    return {k: np.mean(v) for k, v in metrics.items()}


def diagnose_records(data_dir, labels_file, num_samples=5):
    """Diagnose data availability and format issues"""
    print("=" * 60)
    print("DIAGNOSTIC CHECK")
    print("=" * 60)

    # Check labels file
    dfmap = pd.read_csv(labels_file, dtype=str)
    print(f"\n1. Labels file: {labels_file}")
    print(f"   Total records: {len(dfmap)}")
    print(f"   Unique labels: {dfmap['label'].unique()}")
    print(f"   Label counts: {dfmap['label'].value_counts().to_dict()}")

    # Sample some records
    print(f"\n2. Checking sample records (first {num_samples}):")
    for i, row in dfmap.head(num_samples).iterrows():
        rec = row['record']
        label = row['label']
        rec_path = os.path.join(data_dir, rec)

        # Check file existence
        dat_exists = os.path.exists(f"{rec_path}.dat")
        hea_exists = os.path.exists(f"{rec_path}.hea")

        print(f"\n   Record: {rec} (Label: {label})")
        print(f"   .dat exists: {dat_exists}")
        print(f"   .hea exists: {hea_exists}")

        if dat_exists and hea_exists:
            try:
                sig, fields = wfdb.rdsamp(rec_path)
                sig_names = fields['sig_name']
                print(f"   ✓ Successfully loaded!")
                print(f"   Sampling rate: {fields['fs']} Hz")
                print(f"   Signal length: {len(sig)} samples")
                print(f"   Available leads: {sig_names}")

                # Check for required leads
                sig_names_lower = [s.strip().lower() for s in sig_names]
                has_ii = 'ii' in sig_names_lower
                has_iii = 'iii' in sig_names_lower
                has_avf = 'avf' in sig_names_lower
                print(f"   Has lead II: {has_ii}")
                print(f"   Has lead III: {has_iii}")
                print(f"   Has lead aVF: {has_avf}")

            except Exception as e:
                print(f"   ✗ Error loading: {e}")

    print("\n" + "=" * 60)


def main():
    labels_file = os.path.join(DATA_DIR, "PTB_LABELS.csv")

    # Run diagnostics first
    if os.path.exists(labels_file):
        diagnose_records(DATA_DIR, labels_file, num_samples=10)
    else:
        print(f"ERROR: {labels_file} not found!")
        return

    # Ask user to continue
    print("\nDo you want to continue with full processing? (This may take a while)")
    response = input("Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Stopping. Please check the diagnostic output above.")
        return

    # Load records
    records = pd.read_csv(labels_file)['record'].tolist()
    print(f"\n\nProcessing {len(records)} records...")

    X, y, subjects = build_dataset(records, DATA_DIR, LEADS)

    if X.empty or len(y) == 0:
        print("\nERROR: No data was extracted!")
        print("Possible issues:")
        print("  1. Required leads (II, III, aVF) not found in records")
        print("  2. Signal files (.dat/.hea) missing or corrupted")
        print("  3. Path mismatch between PTB_LABELS.csv and actual files")
        return

    print(f"\n✓ Successfully extracted data!")
    print(f"  Dataset shape: {X.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Unique subjects: {len(np.unique(subjects))}")

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    print(f"  After cleaning: {X.shape}")

    # Feature selection
    print("\n\nSelecting features...")
    topk, scores = select_top_k_features(X, y, TOP_K_FEATURES)
    print(f"Top {TOP_K_FEATURES} features:")
    for i, feat in enumerate(topk, 1):
        print(f"  {i}. {feat}: {scores[feat]:.4f}")

    # Evaluation
    print("\n\nEvaluating model...")
    results = evaluate_class_oriented(X, y, topk)
    print(f"\nClass-oriented results (10-fold CV with KNN, K=3):")
    print(f"  AUC:         {results['auc']:.4f}")
    print(f"  Accuracy:    {results['acc'] * 100:.2f}%")
    print(f"  Sensitivity: {results['sen'] * 100:.2f}%")
    print(f"  Specificity: {results['spec'] * 100:.2f}%")
    print(f"  Precision:   {results['prec'] * 100:.2f}%")

    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    X[topk].to_csv(os.path.join(output_dir, "features_topk.csv"), index=False)
    pd.DataFrame({"feature": list(scores.keys()), "gainratio": list(scores.values())}) \
        .sort_values("gainratio", ascending=False) \
        .to_csv(os.path.join(output_dir, "gainratio_scores.csv"), index=False)

    print(f"\n✓ Results saved to {output_dir}/")


if __name__ == "__main__":
    main()