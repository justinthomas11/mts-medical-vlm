"""
Master preprocessing and dataset splitting pipeline for IU X-Ray dataset.
Executes text cleaning, label derivation, image linkage, patient-level splitting,
leakage verification assertions, and cryptographic manifest generation.
"""

import sys
from pathlib import Path

# Add repository root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

from src.data.clean_reports import clean_medical_text, build_target_report, build_clinical_query
from src.data.label_derivation import derive_ground_truth_labels, TARGET_DISEASES


def compute_file_sha256(filepath: Path) -> str:
    """Computes the SHA256 cryptographic hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_pipeline(config_path: str = "configs/data_config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project"]["seed"]
    raw_reports_path = Path(cfg["paths"]["raw_reports_csv"])
    raw_projections_path = Path(cfg["paths"]["raw_projections_csv"])
    images_dir = Path(cfg["paths"]["raw_images_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("--- Starting Preprocessing & Splitting Pipeline ---")

    # 1. Load Raw Data
    df_reports = pd.read_csv(raw_reports_path)
    df_projections = pd.read_csv(raw_projections_path)
    print(f"Loaded raw reports: {len(df_reports)} rows; projections: {len(df_projections)} rows.")

    # 2. Text Cleaning and Target Report Generation
    cleaned_rows = []
    print("Cleaning report text and constructing clinical targets...")
    for _, row in df_reports.iterrows():
        clean_find = clean_medical_text(row.get("findings", ""))
        clean_imp = clean_medical_text(row.get("impression", ""))
        clean_ind = clean_medical_text(row.get("indication", ""))
        clean_comp = clean_medical_text(row.get("comparison", ""))

        target_report, is_valid_target = build_target_report(
            row.get("findings", ""),
            row.get("impression", "")
        )
        clinical_query = build_clinical_query(row.get("indication", ""), cfg["text_processing"]["query_template"])
        labels = derive_ground_truth_labels(row)

        record = {
            "uid": int(row["uid"]),
            "clean_indication": clean_ind,
            "clean_findings": clean_find,
            "clean_impression": clean_imp,
            "clean_comparison": clean_comp,
            "target_report": target_report,
            "is_valid_target": is_valid_target,
            "clinical_query": clinical_query,
            "raw_mesh": str(row.get("MeSH", "")),
            "raw_problems": str(row.get("Problems", ""))
        }
        record.update(labels)
        cleaned_rows.append(record)

    df_cleaned = pd.DataFrame(cleaned_rows)

    # 3. Image Linkage and View Policy
    print("Linking images and enforcing view policy...")
    projections_by_uid = {}
    for _, row in df_projections.iterrows():
        uid = int(row["uid"])
        fname = str(row["filename"])
        proj = str(row["projection"]).strip()
        if uid not in projections_by_uid:
            projections_by_uid[uid] = []
        projections_by_uid[uid].append({"filename": fname, "projection": proj})

    linkage_rows = []
    for _, row in df_cleaned.iterrows():
        uid = int(row["uid"])
        imgs = projections_by_uid.get(uid, [])
        frontal_imgs = [img["filename"] for img in imgs if img["projection"].lower() == "frontal"]
        lateral_imgs = [img["filename"] for img in imgs if img["projection"].lower() == "lateral"]
        all_imgs = [img["filename"] for img in imgs]

        if frontal_imgs:
            primary_img = frontal_imgs[0]
            primary_view = "Frontal"
            has_frontal = True
        elif lateral_imgs:
            primary_img = lateral_imgs[0]
            primary_view = "Lateral"
            has_frontal = False
        else:
            primary_img = None
            primary_view = "None"
            has_frontal = False

        linkage_rows.append({
            "uid": uid,
            "primary_image_filename": primary_img,
            "primary_view": primary_view,
            "has_frontal": has_frontal,
            "num_frontal_images": len(frontal_imgs),
            "num_lateral_images": len(lateral_imgs),
            "num_total_images": len(all_imgs),
            "all_image_filenames": ";".join(all_imgs)
        })

    df_linkage = pd.DataFrame(linkage_rows)
    df_master = pd.merge(df_cleaned, df_linkage, on="uid", how="left")

    # 4. Patient-Level Stratified Splitting
    print("Performing patient-level stratified split (70/10/20)...")
    # Benchmark cohort: patients with valid targets and frontal views
    benchmark_mask = (df_master["is_valid_target"] == True) & (df_master["has_frontal"] == True)
    df_benchmark = df_master[benchmark_mask].copy()

    train_ratio = cfg["splitting"]["train_ratio"]
    val_ratio = cfg["splitting"]["val_ratio"]
    test_ratio = cfg["splitting"]["test_ratio"]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df_benchmark,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=df_benchmark["is_abnormal"]
    )

    # Second split: val vs test (proportional)
    test_share = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_share,
        random_state=seed,
        stratify=temp_df["is_abnormal"]
    )

    # Assign split column
    df_master["split"] = "excluded"
    df_master.loc[~df_master["has_frontal"], "split"] = "excluded_lateral_only"
    df_master.loc[~df_master["is_valid_target"] & df_master["has_frontal"], "split"] = "excluded_empty_target"
    df_master.loc[df_master["uid"].isin(train_df["uid"]), "split"] = "train"
    df_master.loc[df_master["uid"].isin(val_df["uid"]), "split"] = "val"
    df_master.loc[df_master["uid"].isin(test_df["uid"]), "split"] = "test"

    # 5. Verification Assertions
    train_uids = set(df_master[df_master["split"] == "train"]["uid"])
    val_uids = set(df_master[df_master["split"] == "val"]["uid"])
    test_uids = set(df_master[df_master["split"] == "test"]["uid"])

    assert len(train_uids.intersection(val_uids)) == 0, "Patient leakage detected between train and val!"
    assert len(train_uids.intersection(test_uids)) == 0, "Patient leakage detected between train and test!"
    assert len(val_uids.intersection(test_uids)) == 0, "Patient leakage detected between val and test!"
    print("Assertion passed: ZERO patient overlap across train, val, and test splits.")

    # 6. Save Master CSV & Label Matrix
    master_csv_path = Path(cfg["paths"]["processed_master_csv"])
    df_master.to_csv(master_csv_path, index=False)
    print(f"Master processed dataset saved to {master_csv_path} ({len(df_master)} rows).")

    label_cols = ["uid", "split"] + TARGET_DISEASES + ["No_Finding", "is_abnormal"]
    label_matrix_path = Path(cfg["paths"]["label_matrix_csv"])
    df_master[label_cols].to_csv(label_matrix_path, index=False)
    print(f"Label matrix saved to {label_matrix_path}.")

    # 7. Generate Splits Summary Table
    summary_rows = []
    for split_name in ["train", "val", "test", "excluded_lateral_only", "excluded_empty_target", "all"]:
        if split_name == "all":
            sub_df = df_master
        else:
            sub_df = df_master[df_master["split"] == split_name]

        n_pts = len(sub_df)
        if n_pts == 0:
            continue
        n_abnormal = int(sub_df["is_abnormal"].sum())
        n_normal = n_pts - n_abnormal
        abnormal_pct = (n_abnormal / n_pts) * 100

        row_dict = {
            "split": split_name,
            "patient_count": n_pts,
            "normal_count": n_normal,
            "abnormal_count": n_abnormal,
            "abnormal_percentage": round(abnormal_pct, 2)
        }
        for d in TARGET_DISEASES:
            row_dict[d] = int(sub_df[d].sum())
        summary_rows.append(row_dict)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = Path(cfg["paths"]["splits_summary_csv"])
    df_summary.to_csv(summary_path, index=False)
    print(f"Splits summary saved to {summary_path}.")

    # 8. Cryptographic Manifest Generation with SHA256
    print("Computing SHA256 hashes and building splits manifest...")
    manifest = {
        "metadata": {
            "dataset": "Indiana University Chest X-Ray (IU X-Ray)",
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "view_policy": "Frontal primary",
            "total_records": len(df_master),
            "benchmark_records": len(train_df) + len(val_df) + len(test_df)
        },
        "splits": {
            "train": {"patient_count": len(train_df), "records": []},
            "val": {"patient_count": len(val_df), "records": []},
            "test": {"patient_count": len(test_df), "records": []},
            "excluded": {"patient_count": len(df_master[df_master['split'].str.startswith('excluded')]), "records": []}
        }
    }

    # Populate manifest records
    for _, row in df_master.iterrows():
        split_key = row["split"]
        if split_key.startswith("excluded"):
            dest_key = "excluded"
        else:
            dest_key = split_key

        primary_file = row["primary_image_filename"]
        img_hash = None
        if primary_file and (images_dir / primary_file).exists():
            img_hash = compute_file_sha256(images_dir / primary_file)

        record_entry = {
            "uid": int(row["uid"]),
            "split": str(row["split"]),
            "primary_image_filename": primary_file,
            "primary_image_sha256": img_hash,
            "primary_view": str(row["primary_view"]),
            "is_abnormal": int(row["is_abnormal"]),
            "diseases": {d: int(row[d]) for d in TARGET_DISEASES}
        }
        manifest["splits"][dest_key]["records"].append(record_entry)

    manifest_path = Path(cfg["paths"]["splits_manifest_json"])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Splits manifest with file hashes saved to {manifest_path}.")

    print("--- Preprocessing and Splitting Pipeline Complete ---")
    return {
        "train_patients": len(train_df),
        "val_patients": len(val_df),
        "test_patients": len(test_df),
        "excluded_patients": len(df_master) - (len(train_df) + len(val_df) + len(test_df))
    }


if __name__ == "__main__":
    run_pipeline()
