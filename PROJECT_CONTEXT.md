# Complete Project Context: Trustworthy CXR VLM (`trustworthy-cxr-vlm`)

> **Purpose of this Document:**  
> This file contains the complete, exhaustive technical context of the entire project repository (`TrustMedicalVLM` / `mts-medical-vlm`). It documents all architectural and engineering decisions, empirical dataset audits, preprocessing pipelines, data splits, validation studies, codebase structures, test suites, and next roadmap milestones. It is designed to allow any AI assistant (such as Claude) or researcher to immediately understand every single detail without missing context.

---

## 1. Project Identity and Academic Framework

- **Project Title:** Enhancing the Trustworthiness of Vision-Language Models for Chest X-Ray Diagnosis Using RAG, Grad-CAM Explainability, and Uncertainty Estimation, Evaluated with a Composite Medical Trustworthiness Score (MTS)
- **Repository / Workspace Name:** `TrustMedicalVLM` (Git remote: `justinthomas11/mts-medical-vlm`)
- **Institution:** Christ University, Bangalore
- **Program:** M.Tech in Data Science (Master's Dissertation)
- **Author:** Justin Thomas (`justinsthomas2020@gmail.com`)
- **Academic Guide:** Dr. S. A. Sahaaya Arul Mary
- **Project Status:** 
  - **Phase I (Problem Definition & Literature Review):** Complete (30+ papers analyzed; gap identified: no existing work integrates RAG, Grad-CAM, and Uncertainty Estimation under a unified composite evaluation score).
  - **Phase II (Empirical Data Audit, Preprocessing, Splitting & Validation):** 100% Complete and verified via automated test suite.
  - **Phase III (VLM Baseline Selection, S0-S3 Implementation & Evaluation):** S0–S3 implemented, val-tuned and smoke-tested; final FP16 test-split runs pending (see Section 10).
- **Clinical Disclaimer:** Research prototype only. Not an FDA/CE-cleared medical device; not approved for clinical decision-making.

---

## 2. Core Clinical Motivation & Technical Architecture

### 2.1 The Clinical Problem: Four VLM Failure Modes
Modern Vision-Language Models (e.g., LLaVA-Med, CheXagent) generate fluent radiology reports from chest radiographs, but four critical failure modes prevent clinical deployment:
1. **Hallucination:** Generating plausible-sounding findings with no grounding in verifiable image evidence or patient history.
2. **Lack of Explainability (Black Box):** Clinicians cannot visualize which radiographic anatomical regions or image features drove specific diagnostic assertions.
3. **Stale / Static Knowledge:** Closed-world parametric weights fail to incorporate updated clinical guidelines or rare pathological manifestations.
4. **Uncalibrated Overconfidence:** Generating erroneous diagnostic claims with identical high softmax confidence as correct ones.

### 2.2 The Proposed Solution: Tripartite Trust Integration
This dissertation combines three complementary mechanisms into a single pipeline:
1. **Retrieval-Augmented Generation (RAG):** Dynamic retrieval from a curated medical knowledge base (strictly isolated to training data) to ground text generation.
2. **Visual Explainability (Grad-CAM):** Gradient-weighted Class Activation Mapping highlighting influential spatial compartments in the chest radiograph.
3. **Uncertainty Quantification:** Multi-tiered confidence estimation (autoregressive predictive token entropy + semantic multi-sample CheXbert agreement) to flag low-confidence predictions for human radiologist review.

### 2.3 Proposed Medical Trustworthiness Score (MTS)
A proposed composite metric evaluating models along three essential axes:
$$\text{MTS} = w_1 \cdot \text{Diagnostic Performance} + w_2 \cdot \text{Reliability} + w_3 \cdot \text{Explainability}$$
- **Diagnostic Performance:** Micro- and macro-averaged Precision, Recall, F1, and ROC-AUC across 14 standard thoracic conditions.
- **Reliability:** Calibration error (Expected Calibration Error - ECE, Brier score), Hallucination rate, and inference latency.
- **Explainability:** Localization accuracy via the Pointing Game protocol and Saliency Mass Ratio (SMR) evaluated against condition-specific anatomical masks, complemented by automated clinical validity (RadGraph entity/relation agreement).

### 2.4 Stage-Wise Progressive Ablation (S0 to S3)
To rigorously isolate the contribution of each trust module:
- **Stage S0 (Baseline VLM):** Zero-shot / few-shot standalone VLM generating reports directly from images.
- **Stage S1 (+ RAG):** VLM conditioned on retrieved medical literature and exemplar training reports.
- **Stage S2 (+ RAG + Grad-CAM):** VLM + RAG with visual localization heatmaps for predicted pathologies.
- **Stage S3 (+ RAG + Grad-CAM + Uncertainty):** Complete system outputting report, visual explanation, and confidence flags.

```mermaid
flowchart LR
    A[Chest X-ray + Clinical Query] --> B[Vision-Language Model]
    B --> C[RAG Retrieval<br/>Train-only Knowledge Base]
    C --> D[Draft Report Generation]
    D --> E[Grad-CAM Saliency Maps]
    D --> F[Uncertainty Quantification]
    E --> G[Final Trustworthy Report<br/>+ Explanation + Confidence Flag]
    F --> G
    G --> H[Composite MTS Evaluation]
```

---

## 3. Hardware & Execution Environment

- **Host Machine:** Local Windows 11 workstation.
- **Local GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).
- **System Memory:** 24 GB System RAM.
- **Python Runtime:** Python 3.12.9 (C:\Users\Justin\AppData\Local\Programs\Python\Python312\python.exe).
- **Key Installed Packages:** `torch` (CUDA-enabled), `torchvision`, `pandas`, `numpy`, `Pillow`, `scikit-learn`, `pyyaml`, `pytest`, `seaborn`, `matplotlib`.
- **Compute Strategy (DR-001):**
  - Preprocessing, text cleaning, label extraction, dataset splitting, image transforms, and lightweight model evaluation run locally.
  - For full 7B/8B VLM fine-tuning or unquantized FP16 evaluation exceeding 6 GB VRAM, execution seamlessly transitions to remote GPU environments (Google Colab / Kaggle T4/A100 / cloud compute).

---

## 4. Comprehensive Decision Log (DR-001 through DR-026)

Every engineering and clinical choice is formally documented in `docs/decision_log.md`:

| ID | Title | Status | Core Decision & Rationale |
| :--- | :--- | :---: | :--- |
| **DR-001** | Compute Strategy | Accepted | Local RTX 3050 (6GB) for preprocessing, auditing, unit tests, and quantized baselines; cloud GPU offloading planned for heavy 7B+ FP16 inference. |
| **DR-002** | Dataset Source | Accepted | Use verified local Indiana University Chest X-Ray (IU X-Ray) mirror (7,470 images, 2 CSVs); avoids re-downloading and bypasses PhysioNet/CITI credentialing delays. |
| **DR-003** | Automated Clinical Validity Protocol | Accepted | Certified thoracic radiologist panel unavailable for real-time grading; standardized clinical NLP surrogates adopted: RadGraph (entity/relation F1) and CheXbert diagnostic concordance. |
| **DR-004** | Patient-Level Splitting & RAG Isolation | Accepted | IU X-Ray has multiple images per patient (`uid`). Split strictly at patient level (70/10/20, seed 42) to guarantee zero leakage (`assert len(train_uids & test_uids) == 0`). RAG vector database indexed strictly from `train` partition. |
| **DR-005** | Diagnostic Label Derivation | Accepted | IU X-Ray lacks native multi-label binary ground truth. Extracted 14 CheXpert/NIH categories from text and corroborated with curated NLM MeSH/Problems indexation. |
| **DR-006** | Localization Ground Truth Proxy | Accepted | IU X-Ray has no radiologist bounding boxes. Replaced ad-hoc manual boxes with literature-backed Pointing Game protocol and anatomical priors (superseded in detail by DR-016). |
| **DR-007** | Image View Policy | Accepted | Frontal views (PA/AP) designated as primary diagnostic inputs. Lateral projections preserved in metadata to support future multi-view ablation studies. |
| **DR-008** | Target Text & De-identification | Accepted | Concatenate Findings and Impression into structured target (`FINDINGS: ... \nIMPRESSION: ...`). Normalize hospital de-identification placeholders (`XXXX`) into semantic tags (`[DATE]`, `[AGE]`, `[DOCTOR]`, `[REDACTED]`). |
| **DR-009** | Repository Restructuring | Accepted | Flatten nested subdirectory structure directly into workspace root `TrustMedicalVLM` mapped to GitHub repo `mts-medical-vlm`. |
| **DR-010** | Empty/Partial Section Handling | Accepted | If Findings missing, fallback to Impression; if Impression missing, fallback to Findings. If both empty (<10 chars, 23 records), assign to `excluded_empty_target`. Maximizes usable cohort (3,666 patients). |
| **DR-011** | Negation & Multi-Label Rules | Accepted | Sentence-level negation parsing with forward/backward scopes and adversative boundaries (`but`, `however`) combined with MeSH/Problems concept mapping. |
| **DR-012** | Cryptographic Manifest | Accepted | Calculate SHA256 hashes for all primary diagnostic images across all partitions, saved in `data/processed/splits_manifest.json` for verifiable auditability. |
| **DR-013** | Symmetric CheXbert Scoring | Accepted | Score both reference reports and VLM-generated reports using Stanford CheXbert model. Eliminates labeler-induced asymmetric bias between ground truth and generation. Rule-based labeler retained strictly for offline splitting/audit. |
| **DR-014** | Abnormal Rate Reconciliation | Accepted | Reconciled ~8% divergence between raw audit (64.19%) and initial text regex (72.24%). Curated NLM MeSH/Problems tag is authoritative: if tagged "normal", study is strictly normal (`is_abnormal=0`). Regex extraction populates disease matrix only for non-normal studies. Aligns audit and splits to exactly 64.19%. |
| **DR-015** | Uncertainty Quantification Redesign | Accepted | Dropped MC Dropout because modern VLM autoregressive decoders (LLaMA/Vicuna) have no active dropout during inference. Implemented two-tiered protocol: (1) Token Entropy $H(Y\mid X)$ from greedy logits; (2) Semantic consensus across 5 stochastic sampled completions ($T=0.7$) labeled by CheXbert. |
| **DR-016** | Grad-CAM Redesign & Anatomical Grounding | Accepted | Full backprop through 7B LLM requires ~24GB+ VRAM. Redesign: freeze pretrained vision encoder (CLIP ViT-L/14), train lightweight linear classification head on visual tokens using **train split only**, backprop Grad-CAM into vision encoder. Evaluate Pointing Game / SMR against organ-specific masks from `torchxrayvision` PSPNet (Heart for cardiomegaly, Costophrenic angles for effusion, etc.) rather than trivial whole-lung masks. |
| **DR-017** | Evaluation Ontology Standardization | Accepted | Standardize strictly on CheXbert's native 14-observation ontology for model scoring (S0-S3). Retain rule-based 14 NIH-style matrix for offline preprocessing and stratification. |
| **DR-018** | Hybrid RAG Retrieval (S1) | Accepted | Top-3 train reports by α·cos(CLIP image) + (1−α)·cos(MedCPT text); α tuned on val only; train-only index with leakage assertion. |
| **DR-019** | Grad-CAM Targets & Metrics | Accepted | Condition-specific compartments derived from torchxrayvision PSPNet masks; Pointing Game + SMR with chance baseline and Wilson CI; rib-fracture and pneumothorax-count limitations documented. |
| **DR-020** | Review Flag & Calibration (S3) | Accepted | Percentile-fused entropy + sample disagreement; flag threshold = val quantile at a 20% review budget; label-level ECE with confidence 1.0 for S0–S2. |
| **DR-021** | MTS Definition & Weights | Accepted | MTS = 0.4·Diagnostic + 0.3·Reliability + 0.3·Explainability, fixed before test; equal and diagnostic-heavy weights reported as sensitivity. |
| **DR-022** | Checkpoint & Tooling Compatibility | Accepted | HF-format LLaVA-Med v1.5 conversion (4-bit local, FP16 cloud); RadGraph in a transformers<5 venv; NumPy exact search when FAISS is blocked. |
| **DR-023** | Effusion Target Correction | Accepted | Pleural Effusion target = dilated lower third of lungs; PSPNet Facies Diaphragmatica dropped because it marks the sub-diaphragmatic abdomen (amends DR-019; found on train/val). |
| **DR-024** | Prompt Selection (val) | Accepted | Roadmap prompt kept for S0–S3 after a 5-template val comparison; micro-F1 favoured an always-'normal' prompt, so selection used CheXbert macro-F1 (criterion change disclosed). |
| **DR-025** | RAG Prompt Layout (val) | Accepted | Query first, retrieved reports, then an explicit write-this-report instruction; the original layout returned empty reports for 15% of val patients. |
| **DR-026** | Chance-Corrected Explainability | Accepted | MTS explainability = mean of chance-corrected Pointing Game and SMR, so chance-level heatmaps add ~0 (amends DR-021; fixed before test). |

---

## 5. Dataset Audit & Preprocessing Empirical Findings

Conducted via `src/data/audit.py` on 100% of raw tabular and image records (`reports/data_audit.md`, `reports/audit_metrics.json`):

### 5.1 Dataset Demographics & Projections
- **Total Patients (`uid`):** 3,851 unique individuals.
- **Total Projections in CSV (`indiana_projections.csv`):** 7,466 rows (100% linked to valid `uid`s, 0 duplicates).
- **Projections Breakdown:** 3,818 Frontal (51.14%), 3,648 Lateral (48.86%).
- **Images per Patient:** Mean = 1.94, Median = 2.0, Range = 1 to 5:
  - 2 images: 3,210 patients (83.36%)
  - 1 image: 446 patients (11.58%)
  - 3 images: 181 patients (4.70%)
  - 4 images: 13 patients (0.34%)
  - 5 images: 1 patient (0.03%)
- **Patient View Availability:**
  - Both Frontal and Lateral: 3,388 patients (87.98%)
  - Frontal only: 301 patients (7.82%)
  - Lateral only: 162 patients (4.21%) — *Excluded from single-view benchmark cohort (DR-007)*
  - Patients with at least one Frontal view: 3,689 patients (95.79%)

### 5.2 Physical Image Files on Disk (`data/raw/images/images_normalized/`)
- **Total Files on Disk:** 7,470 PNG images.
- **Missing Images:** 0 (All 7,466 filenames referenced in projections CSV exist on disk).
- **Orphan Files on Disk:** 4 files present on disk but unreferenced in CSV:
  - `2084_IM-0715-1001-0002.dcm.png`
  - `2084_IM-0715-2001-0001.dcm.png`
  - `2560_IM-1064-4001.dcm.png`
  - `3809_IM-1919-1003002.dcm.png`
- **Corrupt / Unreadable Files:** 0 (All 7,470 verified readable via PIL).
- **Image Mode:** 100% 8-bit Grayscale (`L` mode).
- **Dimensions:** Width: 1,529 to 2,891 px (mean 2,155.77 px); Height: 1,760 to 3,001 px (mean 2,220.37 px).
- **Intensity:** Mean pixel intensity: 147.39 / 255.0; Standard deviation: 25.80 / 255.0 (sampled over 500 images).

### 5.3 Clinical Text Statistics & De-identification
- **Total Reports in CSV (`indiana_reports.csv`):** 3,851 rows (0 duplicate rows).
- **Missing Columns:** `uid` (0), `MeSH` (0), `Problems` (0), `image` (0), `indication` (86 missing, 2.23%), `comparison` (1,166 missing, 30.28%), `findings` (514 missing, 13.35%), `impression` (31 missing, 0.80%).
- **Both Findings & Impression Missing:** 25 reports (0.65%; 23 records with <10 characters).
- **Text Lengths:**
  - Findings: Mean = 27.26 words, Median = 27.0 words, Max = 169 words.
  - Impression: Mean = 10.48 words, Median = 5.0 words, Max = 130 words.
  - Combined Target: Mean = 37.73 words, Median = 34.0 words, Max = 230 words.
- **De-identification Tokens (`XXXX`):** 
  - 7,118 occurrences across 3,052 reports (79.25% of all reports contain $\ge 1$ placeholder).
  - Average of 1.85 tokens per report.
  - Addressed via regex normalization into `[DATE]`, `[AGE]`, `[DOCTOR]`, `[REDACTED]`.

### 5.4 Diagnostic Balance (Full Cohort of 3,851 Patients)
- **Normal Reports:** 1,379 (35.81%)
- **Abnormal Reports:** 2,472 (64.19%)
- **Imbalance Ratio:** 1.79:1 (Abnormal to Normal).

---

## 6. Dataset Partitioning & Cohort Breakdown

Implemented in `src/data/split_dataset.py`, configured in `configs/data_config.yaml`, and summarized in `data/processed/splits_summary.csv`:

### 6.1 Cohort Definitions
- **Total Patients in Raw Data:** 3,851
- **Benchmark Cohort:** 3,666 patients (possessing valid text target $\ge 10$ characters AND at least one frontal radiograph).
- **Excluded Cohorts:**
  - `excluded_lateral_only`: 162 patients (4.21%) lacking frontal projections.
  - `excluded_empty_target`: 23 patients (0.60%) possessing empty/uninformative findings and impressions.

### 6.2 Patient-Level Partitioning Table (Fixed Seed 42, Stratified by `is_abnormal`)

| Split Name | Patients | % of Benchmark | Normal | Abnormal | Abnormal % | Role in Pipeline |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`train`** | 2,566 | 69.99% | 924 | 1,642 | 63.99% | Sole training corpus for RAG vector index & visual classifier head |
| **`val`** | 366 | 9.98% | 132 | 234 | 63.93% | Prompt tuning, confidence thresholding, hyperparameter selection |
| **`test`** | 734 | 20.02% | 264 | 470 | 64.03% | Held-out evaluation across S0, S1, S2, and S3 ablation stages |
| `excluded_lateral_only` | 162 | N/A | 36 | 126 | 77.78% | Multi-view / lateral extension studies |
| `excluded_empty_target` | 23 | N/A | 23 | 0 | 0.00% | Excluded from text generation |
| **Total All Patients** | **3,851** | **100.0%** | **1,379** | **2,472** | **64.19%** | Master dataset |

### 6.3 Pathology Prevalence Across Partitions (`splits_summary.csv`)

| Condition | Train (2,566) | Val (366) | Test (734) | Excluded Lateral (162) | Excluded Empty (23) | Total Cases (3,851) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Atelectasis** | 369 | 55 | 107 | 38 | 0 | 569 |
| **Cardiomegaly** | 215 | 37 | 76 | 26 | 0 | 354 |
| **Consolidation** | 130 | 25 | 41 | 16 | 0 | 212 |
| **Edema** | 76 | 4 | 29 | 7 | 0 | 116 |
| **Effusion** | 122 | 19 | 29 | 13 | 0 | 183 |
| **Emphysema** | 150 | 24 | 36 | 3 | 0 | 213 |
| **Fibrosis** | 169 | 27 | 38 | 12 | 0 | 246 |
| **Fracture** | 72 | 17 | 13 | 3 | 0 | 105 |
| **Hernia** | 29 | 7 | 12 | 3 | 0 | 51 |
| **Infiltration** | 64 | 12 | 18 | 5 | 0 | 99 |
| **Mass** | 24 | 3 | 9 | 1 | 0 | 37 |
| **Nodule** | 348 | 35 | 105 | 16 | 0 | 504 |
| **Pleural Thickening** | 37 | 9 | 11 | 0 | 0 | 57 |
| **Pneumothorax** | 25 | 2 | 3 | 1 | 0 | 31 |

### 6.4 Data Leakage Verification
Patient isolation across splits is enforced by strict Python set assertions:
```python
assert len(train_uids.intersection(val_uids)) == 0
assert len(train_uids.intersection(test_uids)) == 0
assert len(val_uids.intersection(test_uids)) == 0
```
Verified passing with 0 overlapping patients across all three partitions.

---

## 7. Labeler Validation Study (100-Sample Empirical Audit)

Documented in `reports/labeler_validation_100.md` via `src/data/validate_labeler.py`:

- **Design:** Evaluated 100 randomly sampled IU X-Ray reports (Seed 42) comparing rule-based regex extraction against curated NLM MeSH/Problems tags.
- **Sample Distribution:** 36 Normal (36%), 64 Abnormal (64%).
- **Normal/Abnormal Concordance:** 100.0% (36/36 normal, 64/64 abnormal).
- **Condition-Level Performance on Sample:**
  - Atelectasis: Support 12, F1 = 1.0000
  - Cardiomegaly: Support 11, F1 = 1.0000
  - Consolidation: Support 1, F1 = 0.6667 (1 FP: UID 3680 "could reflect atelectasis or infiltrate")
  - Edema: Support 3, F1 = 0.8571 (1 FP: UID 3328 "questionable interstitial edema")
  - Effusion: Support 3, F1 = 0.8571 (1 FP: UID 1076 "question of pneumonia and/or pleural effusion")
  - Emphysema: Support 4, F1 = 0.7273 (3 FPs: UIDs 2151, 1791, 3533 where "hyperinflation" was flagged as emphysema)
  - Fibrosis: Support 7, F1 = 0.9333 (1 FP: UID 501 "possibly cyst, scarring, or pneumonia")
  - Fracture: Support 3, F1 = 0.8571 (1 FP: UID 3658 "fracture is possible if high energy trauma")
  - Infiltration: Support 0, F1 = 0.0000 (1 FP: UID 3680)
  - Nodule: Support 16, F1 = 1.0000
  - Pleural Thickening: Support 2, F1 = 1.0000
- **Identified Failure Modes:**
  1. *Speculative/Differential Mentions:* Clinicians writing "questionable", "possible", or "cannot exclude" trigger false positives.
  2. *Granular Terminology:* Subtle phrasing ("streaky opacities") missed without exact dictionary matches.
  3. *Modality Recommendations:* "CT scan is sensitive to nodules" triggers false positive for nodule.
- **Direct Scientific Impact:** Led directly to **Decision Record 013** (Symmetric CheXbert scoring) — proving why rule-based extractors must not be used for final model evaluation.

---

## 8. Repository Layout & File Manifest

```text
TrustMedicalVLM/
├── configs/
│   ├── data_config.yaml           # Paths, seed (42), splits, ontology
│   └── experiment_config.yaml     # VLM id, prompt (DR-024), RAG k/alpha/layout (DR-018/025), Grad-CAM head,
│                                  # uncertainty (DR-015/020), MTS weights (DR-021)
├── data/raw, data/processed/      # git-ignored; processed data is rebuilt by src/data/split_dataset.py
├── docs/decision_log.md           # DR-001 … DR-026
├── notebooks/kaggle_s0_s3_generation.ipynb   # FP16 generation for the final runs (Kaggle, 2x T4)
├── reports/                       # Data audit, labeler validation, figures (incl. gradcam_val_examples.png)
├── results/
│   ├── labels/                    # CheXbert labels of all 3,666 reference reports
│   ├── prompt_selection/          # 5 prompt templates x 40 val patients (DR-024)
│   ├── s0/, s1/                   # generations (JSONL), run configs, CheXbert vectors, per-report scores, metrics
│   ├── s2/                        # head training summary, val localization (Pointing Game / SMR)
│   ├── s3/                        # review-flag fits (val only)
│   ├── ablation/                  # S0–S3 tables + bootstrap CIs
│   └── features/                  # cached CLIP / MedCPT features (git-ignored, ~4.3 GB)
├── src/
│   ├── data/                      # audit, cleaning, rule labeler (splitting/audit only), splitting, transforms
│   ├── models/                    # llava_med.py (4-bit/FP16, entropy, sampling), vision_encoder.py, prompts.py
│   ├── rag/                       # MedCPT encoders, train-only hybrid retriever (FAISS or exact NumPy)
│   ├── xai/                       # Grad-CAM head, Grad-CAM, anatomical targets, Pointing Game / SMR
│   ├── uncertainty/               # sample agreement, ECE, val-fitted review flag, risk–coverage
│   ├── evaluation/                # CheXbert scorer, clinical metrics, BLEU/ROUGE/RadGraph, copy overlap,
│   │                              # stage scorer, MTS, ablation builder, paired bootstrap CIs
│   └── pipeline/                  # label/feature extraction, RAG build, generation, head training,
│                                  # localization, prompt selection, split verification, figures
├── tests/                         # pytest suite — must pass before every commit
├── requirements.txt, requirements-radgraph.txt
└── README.md                      # includes the full Phase II run guide
```

---

## 9. Automated Test Suite

`python -m pytest tests/ -v` covers data linkage and leakage (zero shared `uid`s), transforms, CheXbert
binarisation and clinical metrics, BLEU/ROUGE/RadGraph/copy overlap, prompts and RAG layouts, the retriever
(including a train-only assertion on the saved retrieval files), the Grad-CAM head / Grad-CAM / anatomical
targets / localization metrics, uncertainty and the review flag, the stage scorer, MTS (including the config
weights), the ablation builder, bootstrap CIs, the Kaggle notebook and split verification. RadGraph's test runs
only when `RADGRAPH_PYTHON` is set.

---

## 10. Phase III Status (as of 2026-10-02)

### 10.1 Built and verified
| Roadmap step | Status |
|---|---|
| 1. LLaVA-Med | `chaoyinshe/llava-med-v1.5-mistral-7b-hf` (HF conversion, DR-022); 4-bit locally (~4.3 GB VRAM), FP16 on Kaggle |
| 2. CheXbert + clinical metrics | Done; all 3,666 reference reports labelled (`results/labels/`) |
| 3. S0 | Prompt selected on val (DR-024: roadmap prompt kept, by macro-F1); 20-patient smoke tests on val and test |
| 4. S1 | Train-only hybrid retriever (CLIP image + MedCPT text, alpha = 0.5 on val, DR-018); query-first RAG layout (DR-025); smoke-tested |
| 5. S2 | Head on frozen CLIP layer −2 tokens (train only); val localization done; effusion target corrected (DR-023) |
| 6. S3 | Entropy + 5-sample CheXbert agreement; review flag (20% val budget) fitted on a 20-patient val smoke run |
| 7. MTS + ablation | Weights fixed (0.4 / 0.3 / 0.3, DR-021); explainability chance-corrected (DR-026); table builder and paired bootstrap CIs done; end-to-end dry run on val smoke data |

### 10.2 Val findings so far (tuning data — not results)
- Prompt (40 val): explicit FINDINGS/IMPRESSION prompts made LLaVA-Med call most studies normal (one found 0 of 37 abnormal labels); the roadmap prompt had the best macro-F1 (0.111).
- RAG (40 val): the original layout produced empty reports for 15% of patients; the chosen query-first layout produced none but copies heavily from retrieved reports (mean 4-gram copy overlap 0.60 vs 0.002 for S0).
- Grad-CAM head: val macro AUROC 0.745 (best epoch 17).
- Localization (366 val, 124 positive pairs): overall Pointing Game 0.153 and SMR 0.110 against a chance area of 0.109 — chance level overall; only Pleural Effusion is clearly above chance (Pointing Game 0.632 vs chance 0.197).

### 10.3 Remaining
1. Push the repository (Kaggle clones it), then run `notebooks/kaggle_s0_s3_generation.ipynb` in FP16: S0 test, S1 + 5 samples on val, S1 + 5 samples on test (≈ 6–10 GPU hours).
2. Locally: score the outputs with `evaluate_stage.py` (fit the review flag on val, apply it on test), run `eval_localization.py --split test`, then `build_ablation.py` and `bootstrap_ci.py` for the final S0–S3 table.
3. Write up results, including the limitations above (community checkpoint, 4-bit tuning vs FP16 final runs, chance-level Grad-CAM, RAG copying, no rib masks, 3 Pneumothorax test cases).

No test-split metric has been computed for any stage yet.
