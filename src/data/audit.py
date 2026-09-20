"""
Comprehensive Data Audit Script for Indiana University Chest X-Ray (IU X-Ray) Dataset.
Computes exact dataset statistics, checks image integrity, analyzes text distributions,
and generates diagnostic figures for reports/data_audit.md.
"""

import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def run_audit(config_path: str = "configs/data_config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    reports_path = Path(cfg["paths"]["raw_reports_csv"])
    projections_path = Path(cfg["paths"]["raw_projections_csv"])
    images_dir = Path(cfg["paths"]["raw_images_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("--- Starting Data Audit ---")
    metrics = {}

    # 1. Reports CSV Schema and Integrity
    df_reports = pd.read_csv(reports_path)
    metrics["reports"] = {
        "total_rows": int(len(df_reports)),
        "columns": list(df_reports.columns),
        "dtypes": {col: str(dtype) for col, dtype in df_reports.dtypes.items()},
        "missing_per_col": {col: int(df_reports[col].isna().sum()) for col in df_reports.columns},
        "duplicate_rows": int(df_reports.duplicated().sum()),
        "unique_uids": int(df_reports["uid"].nunique()),
        "duplicate_uids": int(df_reports["uid"].duplicated().sum()),
    }
    print(f"Loaded reports: {len(df_reports)} rows, {df_reports['uid'].nunique()} unique uids.")

    # 2. Projections CSV Schema and Integrity
    df_projections = pd.read_csv(projections_path)
    metrics["projections"] = {
        "total_rows": int(len(df_projections)),
        "columns": list(df_projections.columns),
        "dtypes": {col: str(dtype) for col, dtype in df_projections.dtypes.items()},
        "missing_per_col": {col: int(df_projections[col].isna().sum()) for col in df_projections.columns},
        "duplicate_rows": int(df_projections.duplicated().sum()),
        "unique_uids": int(df_projections["uid"].nunique()),
        "duplicate_uids": int(df_projections["uid"].duplicated().sum()),
        "unique_filenames": int(df_projections["filename"].nunique()),
    }
    print(f"Loaded projections: {len(df_projections)} rows, {df_projections['uid'].nunique()} unique uids.")

    # 3. Patient & View Distribution
    projection_counts = df_projections["projection"].value_counts().to_dict()
    metrics["projections"]["view_counts"] = {str(k): int(v) for k, v in projection_counts.items()}

    images_per_patient = df_projections.groupby("uid")["filename"].count()
    metrics["patient_views"] = {
        "images_per_patient_min": int(images_per_patient.min()),
        "images_per_patient_max": int(images_per_patient.max()),
        "images_per_patient_mean": float(images_per_patient.mean()),
        "images_per_patient_median": float(images_per_patient.median()),
        "images_per_patient_distribution": {str(k): int(v) for k, v in images_per_patient.value_counts().to_dict().items()}
    }

    # Analyze per-patient view availability
    views_per_patient = df_projections.groupby("uid")["projection"].apply(lambda s: set(s.dropna().str.strip()))
    has_frontal = views_per_patient.apply(lambda v: any(x.lower() == "frontal" for x in v))
    has_lateral = views_per_patient.apply(lambda v: any(x.lower() == "lateral" for x in v))

    patients_both = int((has_frontal & has_lateral).sum())
    patients_frontal_only = int((has_frontal & ~has_lateral).sum())
    patients_lateral_only = int((~has_frontal & has_lateral).sum())
    patients_neither = int((~has_frontal & ~has_lateral).sum())

    metrics["patient_view_availability"] = {
        "both_frontal_and_lateral": patients_both,
        "frontal_only": patients_frontal_only,
        "lateral_only": patients_lateral_only,
        "neither_frontal_nor_lateral": patients_neither,
        "patients_with_no_frontal": patients_lateral_only + patients_neither
    }
    print(f"View breakdown: Both: {patients_both}, Frontal-only: {patients_frontal_only}, Lateral-only: {patients_lateral_only}")

    # 4. Image-Report Linkage & Disk Integrity
    report_uids = set(df_reports["uid"])
    projection_uids = set(df_projections["uid"])

    uids_in_both = len(report_uids.intersection(projection_uids))
    reports_without_projections = len(report_uids - projection_uids)
    projections_without_reports = len(projection_uids - report_uids)

    disk_files = set(os.listdir(images_dir))
    csv_filenames = set(df_projections["filename"].dropna())

    missing_from_disk = list(csv_filenames - disk_files)
    orphan_files_on_disk = list(disk_files - csv_filenames)

    print(f"Disk check: {len(disk_files)} files on disk. Missing from disk: {len(missing_from_disk)}, Orphans on disk: {len(orphan_files_on_disk)}")

    # Check unreadable / corrupt images
    corrupt_files = []
    image_stats_list = []

    print("Checking image readability and dimensions on disk...")
    for idx, fname in enumerate(disk_files):
        fpath = images_dir / fname
        try:
            with Image.open(fpath) as img:
                w, h = img.size
                mode = img.mode
                # Sample 500 images for intensity statistics to ensure fast audit
                if idx < 500:
                    arr = np.array(img)
                    mean_val = float(np.mean(arr))
                    std_val = float(np.std(arr))
                    min_val = float(np.min(arr))
                    max_val = float(np.max(arr))
                else:
                    mean_val, std_val, min_val, max_val = None, None, None, None
                image_stats_list.append({
                    "filename": fname,
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val
                })
        except Exception as e:
            corrupt_files.append({"filename": fname, "error": str(e)})

    df_img_stats = pd.DataFrame(image_stats_list)

    modes_count = df_img_stats["mode"].value_counts().to_dict()
    widths = df_img_stats["width"]
    heights = df_img_stats["height"]

    sample_intensity = df_img_stats["mean"].dropna()

    metrics["image_integrity"] = {
        "files_on_disk": len(disk_files),
        "files_in_csv": len(csv_filenames),
        "missing_from_disk_count": len(missing_from_disk),
        "missing_from_disk_examples": missing_from_disk[:5],
        "orphan_files_count": len(orphan_files_on_disk),
        "orphan_files_examples": orphan_files_on_disk[:5],
        "corrupt_files_count": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "modes": {str(k): int(v) for k, v in modes_count.items()},
        "width_min": int(widths.min()),
        "width_max": int(widths.max()),
        "width_mean": float(widths.mean()),
        "height_min": int(heights.min()),
        "height_max": int(heights.max()),
        "height_mean": float(heights.mean()),
        "intensity_sample_count": len(sample_intensity),
        "intensity_sample_mean": float(sample_intensity.mean()) if len(sample_intensity) > 0 else None,
        "intensity_sample_std": float(sample_intensity.std()) if len(sample_intensity) > 0 else None
    }

    # 5. Report Text Statistics and De-identification
    findings = df_reports["findings"].fillna("").astype(str)
    impression = df_reports["impression"].fillna("").astype(str)

    findings_char_len = findings.str.len()
    findings_word_len = findings.str.split().str.len()
    impression_char_len = impression.str.len()
    impression_word_len = impression.str.split().str.len()

    combined_text = (findings + " " + impression).str.strip()
    combined_word_len = combined_text.str.split().str.len()

    empty_findings = int((findings.str.strip() == "").sum())
    empty_impression = int((impression.str.strip() == "").sum())
    both_empty = int(((findings.str.strip() == "") & (impression.str.strip() == "")).sum())

    # De-identification token analysis: regex for XXXX tokens
    xxxx_pattern = re.compile(r"\bX+\b", re.IGNORECASE)

    def count_xxxx(text: str) -> int:
        return len(xxxx_pattern.findall(text))

    reports_xxxx_counts = df_reports.apply(
        lambda row: count_xxxx(str(row["findings"])) + count_xxxx(str(row["impression"])) + count_xxxx(str(row["indication"])),
        axis=1
    )
    reports_with_xxxx = int((reports_xxxx_counts > 0).sum())
    total_xxxx_tokens = int(reports_xxxx_counts.sum())

    metrics["report_text"] = {
        "findings_char_len_mean": float(findings_char_len.mean()),
        "findings_word_len_mean": float(findings_word_len.mean()),
        "findings_word_len_median": float(findings_word_len.median()),
        "findings_word_len_max": int(findings_word_len.max()),
        "impression_word_len_mean": float(impression_word_len.mean()),
        "impression_word_len_median": float(impression_word_len.median()),
        "impression_word_len_max": int(impression_word_len.max()),
        "combined_word_len_mean": float(combined_word_len.mean()),
        "combined_word_len_median": float(combined_word_len.median()),
        "combined_word_len_max": int(combined_word_len.max()),
        "empty_findings_count": empty_findings,
        "empty_impression_count": empty_impression,
        "both_empty_count": both_empty,
        "reports_with_deid_tokens": reports_with_xxxx,
        "reports_with_deid_pct": float(reports_with_xxxx / len(df_reports) * 100),
        "total_deid_tokens": total_xxxx_tokens,
        "avg_deid_tokens_per_report": float(total_xxxx_tokens / len(df_reports))
    }

    # 6. Label Landscape: MeSH and Problems
    mesh_series = df_reports["MeSH"].fillna("").astype(str)
    problems_series = df_reports["Problems"].fillna("").astype(str)

    is_normal_mesh = mesh_series.str.strip().str.lower() == "normal"
    is_normal_prob = problems_series.str.strip().str.lower() == "normal"
    is_normal_combined = is_normal_mesh | is_normal_prob

    normal_count = int(is_normal_combined.sum())
    abnormal_count = int((~is_normal_combined).sum())

    # Tag parsing
    all_mesh_tags = []
    for m in mesh_series:
        if m.strip() and m.strip().lower() != "normal":
            tags = [t.strip() for t in m.split(";") if t.strip()]
            all_mesh_tags.extend(tags)

    mesh_tag_counts = pd.Series(all_mesh_tags).value_counts()

    all_prob_tags = []
    for p in problems_series:
        if p.strip() and p.strip().lower() != "normal":
            tags = [t.strip() for t in p.split(";") if t.strip()]
            all_prob_tags.extend(tags)

    prob_tag_counts = pd.Series(all_prob_tags).value_counts()

    metrics["labels"] = {
        "normal_reports": normal_count,
        "abnormal_reports": abnormal_count,
        "normal_percentage": float(normal_count / len(df_reports) * 100),
        "abnormal_percentage": float(abnormal_count / len(df_reports) * 100),
        "top_15_mesh_tags": {str(k): int(v) for k, v in mesh_tag_counts.head(15).items()},
        "top_15_problem_tags": {str(k): int(v) for k, v in prob_tag_counts.head(15).items()}
    }

    # 7. Generate Figures

    # Figure 1: Projections & Images Per Patient
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_projections["projection"].value_counts().plot(kind="bar", ax=axes[0], color="#2b5c8f", edgecolor="black")
    axes[0].set_title("Image Projection Distribution")
    axes[0].set_xlabel("Projection")
    axes[0].set_ylabel("Number of Images")
    axes[0].tick_params(axis="x", rotation=0)

    images_per_patient.value_counts().sort_index().plot(kind="bar", ax=axes[1], color="#4c9f70", edgecolor="black")
    axes[1].set_title("Images per Patient Distribution")
    axes[1].set_xlabel("Number of Images")
    axes[1].set_ylabel("Patient Count")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig_path_1 = figures_dir / "views_distribution.png"
    plt.savefig(fig_path_1, dpi=300)
    plt.close()

    # Figure 2: Text Length Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(findings_word_len, bins=35, ax=axes[0], color="#2b5c8f", kde=True, label="Findings")
    sns.histplot(impression_word_len, bins=35, ax=axes[0], color="#d95f02", kde=True, label="Impression")
    axes[0].set_title("Report Word Count Distributions")
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Report Frequency")
    axes[0].set_xlim(0, 100)
    axes[0].legend()

    sns.boxplot(data=[findings_word_len, impression_word_len, combined_word_len], ax=axes[1], palette="Blues")
    axes[1].set_xticklabels(["Findings", "Impression", "Combined Target"])
    axes[1].set_title("Word Count Summary Boxplots")
    axes[1].set_ylabel("Word Count")
    axes[1].set_ylim(0, 120)
    plt.tight_layout()
    fig_path_2 = figures_dir / "text_lengths.png"
    plt.savefig(fig_path_2, dpi=300)
    plt.close()

    # Figure 3: Label Landscape
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].pie([normal_count, abnormal_count], labels=["Normal", "Abnormal"], autopct="%1.1f%%", colors=["#4c9f70", "#d95f02"], startangle=140, explode=(0.05, 0))
    axes[0].set_title("Diagnostic Distribution (Normal vs Abnormal)")

    prob_tag_counts.head(10).plot(kind="barh", ax=axes[1], color="#2b5c8f", edgecolor="black")
    axes[1].invert_yaxis()
    axes[1].set_title("Top 10 Pathology Tags (Problems Field)")
    axes[1].set_xlabel("Occurrences")
    plt.tight_layout()
    fig_path_3 = figures_dir / "label_distribution.png"
    plt.savefig(fig_path_3, dpi=300)
    plt.close()

    # Figure 4: Image Dimensions and Intensity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df_img_stats["width"], bins=20, ax=axes[0], color="#2b5c8f", kde=False)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Pixels")
    axes[0].set_ylabel("Image Count")

    sns.histplot(sample_intensity, bins=25, ax=axes[1], color="#7570b3", kde=True)
    axes[1].set_title("Mean Pixel Intensity (Sample of 500 Images)")
    axes[1].set_xlabel("Mean 8-bit Pixel Value (0-255)")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    fig_path_4 = figures_dir / "image_stats.png"
    plt.savefig(fig_path_4, dpi=300)
    plt.close()

    # Save metrics JSON
    metrics_path = Path("reports/audit_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Data audit completed successfully. Metrics written to reports/audit_metrics.json and figures saved.")
    return metrics


if __name__ == "__main__":
    run_audit()
