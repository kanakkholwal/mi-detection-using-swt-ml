#!/usr/bin/env python3
"""
Fixed: Sharma & Sunkaria (2018) - SWT + features + KNN/SVM on PTB-DB.
Optimized for speed with parallel processing and fast entropy approximation.
"""

import math
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool, cpu_count

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
N_JOBS = max(1, cpu_count() - 1)  # Use all but one CPU core


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
    """Ultra-fast approximate sample entropy"""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    # Aggressively downsample for entropy calculation
    # Entropy measures complexity, which is preserved even with fewer samples
    if N > 100:
        step = max(1, N // 100)
        x = x[::step]
        N = len(x)

    r = r_factor * np.std(x)
    if r == 0 or N < m + 2:
        return 0.0

    # Only sample a subset of comparisons for speed
    max_comparisons = min(500, (N - m) * (N - m - 1) // 2)

    B_count = 0
    A_count = 0
    comparisons_done = 0

    # Stratified sampling: sample uniformly across the signal
    i_step = max(1, (N - m) // 20)  # Sample ~20 starting points

    for i in range(0, N - m, i_step):
        template_m = x[i:i + m]

        # Compare with sampled subsequent templates
        j_step = max(1, (N - m - i) // 10)  # Sample ~10 comparisons per i
        for j in range(i + 1, N - m + 1, j_step):
            if comparisons_done >= max_comparisons:
                break

            dist_m = np.max(np.abs(template_m - x[j:j + m]))
            if dist_m <= r:
                B_count += 1

                if i < N - m - 1 and j < N - m:
                    dist_m1 = np.max(np.abs(x[i:i + m + 1] - x[j:j + m + 1]))
                    if dist_m1 <= r:
                        A_count += 1

            comparisons_done += 1

        if comparisons_done >= max_comparisons:
            break

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


def process_single_segment(args):
    """Process a single segment - designed for parallel processing"""
    seg_data, leads = args
    features = {}

    for li, lead in enumerate(leads):
        seg = seg_data[li]

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

    return features


def build_dataset(records, data_dir=DATA_DIR, leads=None, use_parallel=True):
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
        except Exception:
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

        # Process each lead - preprocessing
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

        # Prepare segment data for parallel processing
        nseg = min(len(s) for s in per_lead_segments)
        segment_args = []
        for si in range(nseg):
            seg_data = [per_lead_segments[li][si] for li in range(len(leads))]
            segment_args.append((seg_data, leads))

        # Process segments (parallel or sequential)
        if use_parallel and len(segment_args) > 4:
            with Pool(processes=min(N_JOBS, len(segment_args))) as pool:
                segment_features = pool.map(process_single_segment, segment_args)
        else:
            segment_features = [process_single_segment(arg) for arg in segment_args]

        # Collect results
        for features in segment_features:
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


def save_and_visualize_results(X, y, topk, scores, results, output_base="results"):
    """Save and visualize all results with timestamp-based organization"""
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving results to: {output_dir}/")

    # 1. Save top-k features CSV
    X[topk].to_csv(os.path.join(output_dir, "features_topk.csv"), index=False)

    # 2. Save gain ratio scores
    gain_df = pd.DataFrame({
        "feature": list(scores.keys()),
        "gainratio": list(scores.values())
    }).sort_values("gainratio", ascending=False)
    gain_df.to_csv(os.path.join(output_dir, "gainratio_scores.csv"), index=False)

    # 3. Save classification results
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(output_dir, "classification_results.csv"), index=False)

    # === VISUALIZATIONS ===
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100

    # 4. Top 10 Features Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    top10_df = gain_df.head(10).sort_values("gainratio", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top10_df)))
    ax.barh(range(len(top10_df)), top10_df['gainratio'], color=colors)
    ax.set_yticks(range(len(top10_df)))
    ax.set_yticklabels(top10_df['feature'], fontsize=9)
    ax.set_xlabel('Gain Ratio', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Features by Gain Ratio', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top10_features.png"), bbox_inches='tight')
    plt.close()

    # 5. Classification Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision']
    metrics_values = [
        results['auc'],
        results['acc'],
        results['sen'],
        results['spec'],
        results['prec']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.4f}\n({val * 100:.2f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Classification Performance Metrics (10-Fold CV)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_metrics.png"), bbox_inches='tight')
    plt.close()

    # 6. Feature Distribution Analysis (Top 3 features)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, feat in enumerate(topk[:3]):
        ax = axes[idx]
        hc_data = X.loc[y == 0, feat]
        imi_data = X.loc[y == 1, feat]

        ax.hist(hc_data, bins=30, alpha=0.6, label='HC', color='blue', edgecolor='black')
        ax.hist(imi_data, bins=30, alpha=0.6, label='IMI', color='red', edgecolor='black')
        ax.set_xlabel('Feature Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{feat}\n(Gain Ratio: {scores[feat]:.4f})', fontsize=10, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Distribution of Top 3 Features', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top3_distributions.png"), bbox_inches='tight')
    plt.close()

    # 7. Gain Ratio Scores - Full Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    top30 = gain_df.head(30)
    ax.plot(range(len(top30)), top30['gainratio'], marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax.fill_between(range(len(top30)), top30['gainratio'], alpha=0.3, color='#2E86AB')
    ax.axvline(x=9.5, color='red', linestyle='--', linewidth=2, label='Top 10 cutoff')
    ax.set_xlabel('Feature Rank', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gain Ratio', fontsize=11, fontweight='bold')
    ax.set_title('Gain Ratio Distribution (Top 30 Features)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gainratio_distribution.png"), bbox_inches='tight')
    plt.close()

    # 8. Summary Text Report
    with open(os.path.join(output_dir, "summary_report.txt"), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MYOCARDIAL INFARCTION DETECTION - RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset Size: {len(X)} segments\n")
        f.write(f"HC samples: {np.sum(y == 0)}\n")
        f.write(f"IMI samples: {np.sum(y == 1)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("TOP 10 FEATURES (by Gain Ratio)\n")
        f.write("-" * 70 + "\n")
        for i, (feat, score) in enumerate(gain_df.head(10).values, 1):
            f.write(f"{i:2d}. {feat:40s} : {score:.6f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("CLASSIFICATION RESULTS (10-Fold Cross-Validation with KNN, K=3)\n")
        f.write("-" * 70 + "\n")
        f.write(f"AUC (ROC):        {results['auc']:.4f}\n")
        f.write(f"Accuracy:         {results['acc']:.4f} ({results['acc'] * 100:.2f}%)\n")
        f.write(f"Sensitivity:      {results['sen']:.4f} ({results['sen'] * 100:.2f}%)\n")
        f.write(f"Specificity:      {results['spec']:.4f} ({results['spec'] * 100:.2f}%)\n")
        f.write(f"Precision:        {results['prec']:.4f} ({results['prec'] * 100:.2f}%)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Paper Reference: Sharma & Sunkaria (2018)\n")
        f.write("Paper Results (Class-oriented with KNN):\n")
        f.write("  ROC=0.9945, Se%=98.67, Sp%=98.72, +P%=98.79, Ac%=98.69\n")
        f.write("=" * 70 + "\n")

    # 9. HTML Dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MI Detection Results - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 32px; font-weight: bold; }}
            .metric-label {{ font-size: 14px; margin-top: 5px; opacity: 0.9; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
            .info-box {{ background: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Myocardial Infarction Detection Results</h1>
            <p><strong>Generated:</strong> {timestamp}</p>

            <div class="info-box">
                <strong>Dataset Information:</strong><br>
                Total Segments: {len(X)} | HC: {np.sum(y == 0)} | IMI: {np.sum(y == 1)}
            </div>

            <h2>Performance Metrics (10-Fold Cross-Validation)</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{results['auc']:.4f}</div>
                    <div class="metric-label">AUC (ROC)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results['acc'] * 100:.2f}%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results['sen'] * 100:.2f}%</div>
                    <div class="metric-label">Sensitivity</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results['spec'] * 100:.2f}%</div>
                    <div class="metric-label">Specificity</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results['prec'] * 100:.2f}%</div>
                    <div class="metric-label">Precision</div>
                </div>
            </div>

            <img src="classification_metrics.png" alt="Classification Metrics">

            <h2>Top 10 Selected Features</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Gain Ratio</th>
                </tr>
    """

    for i, (feat, score) in enumerate(gain_df.head(10).values, 1):
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feat}</td>
                    <td>{score:.6f}</td>
                </tr>
        """

    html_content += f"""
            </table>

            <img src="top10_features.png" alt="Top 10 Features">
            <img src="gainratio_distribution.png" alt="Gain Ratio Distribution">

            <h2>Feature Distributions</h2>
            <img src="top3_distributions.png" alt="Top 3 Feature Distributions">

            <h2>Reference</h2>
            <div class="info-box">
                <strong>Paper:</strong> Sharma & Sunkaria (2018) - Inferior myocardial infarction detection using stationary wavelet transform and machine learning approach<br>
                <strong>Paper Results (Class-oriented KNN):</strong> ROC=0.9945, Se%=98.67, Sp%=98.72, +P%=98.79, Ac%=98.69
            </div>
        </div>
    </body>
    </html>
    """

    with open(os.path.join(output_dir, "dashboard.html"), 'w') as f:
        f.write(html_content)

    print(f"\n{'=' * 70}")
    print("RESULTS SAVED SUCCESSFULLY")
    print(f"{'=' * 70}")
    print(f"Location: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  - features_topk.csv          (Top {len(topk)} feature values)")
    print(f"  - gainratio_scores.csv       (All feature scores)")
    print(f"  - classification_results.csv (Performance metrics)")
    print(f"  - summary_report.txt         (Text summary)")
    print(f"  - dashboard.html             (Interactive HTML report)")
    print(f"  - *.png                      (Visualization charts)")
    print(f"\nOpen dashboard.html in your browser for an interactive view!")
    print(f"{'=' * 70}\n")

    return output_dir


def main():
    labels_file = os.path.join(DATA_DIR, "PTB_LABELS.csv")

    # Run diagnostics first
    if os.path.exists(labels_file):
        print("\nRunning diagnostics...")
        diagnose_records(DATA_DIR, labels_file, num_samples=5)
    else:
        print(f"ERROR: {labels_file} not found!")
        return

    # Ask user to continue
    print("\nDo you want to continue with full processing?")
    response = input("Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Stopping. Please check the diagnostic output above.")
        return

    # Load records
    records = pd.read_csv(labels_file)['record'].tolist()
    print(f"\n\nProcessing {len(records)} records...")
    print(f"Using {N_JOBS} CPU cores for parallel processing\n")

    X, y, subjects = build_dataset(records, DATA_DIR, LEADS, use_parallel=True)

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
    print("\nCleaning data...")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    print(f"  After cleaning: {X.shape}")

    # Feature selection
    print("\n\nSelecting features (this may take a few minutes)...")
    topk, scores = select_top_k_features(X, y, TOP_K_FEATURES)
    print(f"\nTop {TOP_K_FEATURES} features:")
    for i, feat in enumerate(topk, 1):
        print(f"  {i:2d}. {feat:40s} : {scores[feat]:.6f}")

    # Evaluation
    print("\n\nEvaluating model (10-fold cross-validation)...")
    results = evaluate_class_oriented(X, y, topk)
    print(f"\nClass-oriented results (KNN, K=3):")
    print(f"  AUC:         {results['auc']:.4f}")
    print(f"  Accuracy:    {results['acc'] * 100:.2f}%")
    print(f"  Sensitivity: {results['sen'] * 100:.2f}%")
    print(f"  Specificity: {results['spec'] * 100:.2f}%")
    print(f"  Precision:   {results['prec'] * 100:.2f}%")

    # Save and visualize results
    print("\n\nGenerating visualizations and saving results...")
    output_dir = save_and_visualize_results(X, y, topk, scores, results)

    # Final comparison with paper
    print("\n" + "=" * 70)
    print("COMPARISON WITH PAPER RESULTS")
    print("=" * 70)
    print("\nPaper (Sharma & Sunkaria 2018) - Class-oriented KNN (K=3):")
    print("  ROC:         0.9945")
    print("  Sensitivity: 98.67%")
    print("  Specificity: 98.72%")
    print("  Precision:   98.79%")
    print("  Accuracy:    98.69%")
    print("\nYour Replication:")
    print(f"  AUC:         {results['auc']:.4f}")
    print(f"  Sensitivity: {results['sen'] * 100:.2f}%")
    print(f"  Specificity: {results['spec'] * 100:.2f}%")
    print(f"  Precision:   {results['prec'] * 100:.2f}%")
    print(f"  Accuracy:    {results['acc'] * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
