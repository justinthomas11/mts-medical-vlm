# Dataset Card: Indiana University Chest X-Ray (Processed for Trustworthy VLM)

## 1. Dataset Summary
- Dataset Name: Indiana University Chest X-Ray (IU X-Ray) Processed Cohort
- Primary Source: OpenI / National Library of Medicine (Kaggle mirror)
- Task: Vision-Language Model (VLM) Chest Radiograph Diagnostic Generation, Multi-Label Pathology Classification, Explainability, and Uncertainty Calibration
- Total Patients: 3,851 unique patient studies
- Primary Benchmark Cohort: 3,666 patients (possessing valid clinical reports and at least one frontal projection)
- Excluded Cohorts: 162 patients with lateral-only views (preserved in metadata for lateral studies); 23 patients with empty target text

## 2. Dataset Structure and Schema

### 2.1 File Descriptions
- `data/processed/iu_xray_processed.csv`: Cleaned master tabular dataset with normalized text, image linkages, derived pathology labels, and split assignments.
- `data/processed/splits_manifest.json`: Cryptographic manifest containing SHA256 hashes and file paths per split.
- `data/processed/splits_summary.csv`: Aggregated demographic and pathology prevalence across train, val, and test splits.
- `data/processed/label_matrix.csv`: 14-condition multi-label binary ground truth matrix per study.

### 2.2 Feature Specifications (`iu_xray_processed.csv`)
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `uid` | int | Unique patient examination identifier |
| `clean_indication` | str | Normalized reason for exam / symptoms |
| `clean_findings` | str | Normalized radiological findings text |
| `clean_impression` | str | Normalized diagnostic impression text |
| `target_report` | str | Combined generation target (`FINDINGS: ... \nIMPRESSION: ...`) |
| `is_valid_target` | bool | True if target text contains at least 10 characters |
| `clinical_query` | str | Standardized model prompt input |
| `primary_image_filename`| str | Filename of primary diagnostic radiograph on disk |
| `primary_view` | str | Radiographic projection of primary image (`Frontal` or `Lateral`) |
| `has_frontal` | bool | Indicates whether patient study includes a frontal radiograph |
| `num_total_images` | int | Total radiographs associated with patient study |
| `all_image_filenames` | str | Semicolon-delimited list of all images for the patient |
| `split` | str | Partition assignment (`train`, `val`, `test`, `excluded_lateral_only`, `excluded_empty_target`) |
| `is_abnormal` | int | Binary indicator (1 if abnormal finding present, 0 if normal) |
| `No_Finding` | int | Binary indicator (1 if study is normal, 0 if pathology detected) |
| `[14 Pathology Cols]` | int | Binary indicators across 14 CheXpert/NIH conditions |

## 3. Data Splits and Leakage Prevention
To prevent data contamination, partitioning is strictly patient-level (`uid`) with zero patient overlap across splits:
- `train`: 2,566 patients (69.99% of benchmark cohort; 72.21% abnormal prevalence)
- `val`: 366 patients (9.98% of benchmark cohort; 72.40% abnormal prevalence)
- `test`: 734 patients (20.02% of benchmark cohort; 72.21% abnormal prevalence)
- Zero patient overlap verified via automated assertions (`assert len(train_uids & test_uids) == 0`).
- Strict RAG Isolation: Only reports belonging to the `train` partition are indexed into the retrieval knowledge base.

## 4. Pathology Label Ontology
Multi-label binary ground truth labels are extracted across 14 standard thoracic categories:
1. Atelectasis (569 total cases)
2. Cardiomegaly (355 total cases)
3. Consolidation (250 total cases)
4. Edema (129 total cases)
5. Effusion (571 total cases)
6. Emphysema (213 total cases)
7. Fibrosis (246 total cases)
8. Fracture (108 total cases)
9. Hernia (52 total cases)
10. Infiltration (101 total cases)
11. Mass (87 total cases)
12. Nodule (554 total cases)
13. Pleural Thickening (58 total cases)
14. Pneumothorax (247 total cases)

## 5. Text Preprocessing and De-identification
- Hospital de-identification placeholders (`XXXX`) are normalized to standardized semantic tokens: `[DATE]`, `[AGE]`, `[DOCTOR]`, and `[REDACTED]`.
- Punctuation spacing and whitespace are normalized.
- Missing sections are handled with graceful fallbacks (Impression-only target when Findings are absent).

## 6. Image Transformation Standards
- Physical images are 8-bit grayscale radiographs with resolutions between 1529x1760 and 2891x3001 pixels.
- Transform module (`src.data.transforms.MedicalImageTransform`) supports:
  - Aspect-ratio preserving letterboxing to target dimensions (default: 512x512).
  - Conversion to 3-channel RGB for VLM vision encoders.
  - Standard ImageNet channel normalization.
  - Fully compatible with PyTorch `Dataset` and `DataLoader`.

## 7. Limitations
- IU X-Ray does not provide radiologist-annotated bounding boxes; spatial explainability must use anatomical priors and Pointing Game evaluations.
- 162 patients lack frontal radiographs and cannot be evaluated with single-view frontal baselines without multi-view adaptations.
