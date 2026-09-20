# Architecture and Engineering Decision Log

Project: Enhancing Trustworthiness of Vision-Language Models for Medical Image Diagnosis
Institution: Christ University, Bangalore (M.Tech Data Science Dissertation)

---

## Decision Record 001: Execution Environment and Compute Strategy
- Date: 2026-09-20
- Status: Accepted
- Context: Local development machine possesses an NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM), 24 GB System RAM, running Windows 11 with Python 3.12.9.
- Decision: Perform all data audits, text cleaning, label derivations, dataset splits, and quantized/small baseline VLM evaluations locally. Plan for remote execution (Google Colab / Kaggle GPU / cloud compute) if Phase II VLM architectures (LLaVA-Med, CheXagent) exceed 6 GB VRAM during full FP16 or multi-stage inference.
- Alternatives Considered:
  1. Full cloud-only workflow: Rejected due to slower local iteration and upload latency for dataset inspection.
  2. Local-only without offloading option: Rejected due to 6 GB VRAM constraint on 7B parameter models without heavy 4-bit quantization.
- Justification: Enables rapid, zero-cost local preprocessing and validation while preserving scalability for heavier VLM stages.

---

## Decision Record 002: Dataset Source and Integrity
- Date: 2026-09-20
- Status: Accepted
- Context: Indiana University Chest X-Ray dataset (IU X-Ray) local mirror containing 7,470 normalized images, `indiana_projections.csv`, and `indiana_reports.csv`.
- Decision: Use the verified local files stored in `data/raw/` without re-downloading from Kaggle.
- Alternatives Considered:
  1. Re-downloading from OpenI / NLM / Kaggle: Rejected because verified local files already exist in workspace.
  2. Switching to MIMIC-CXR: Ruled out during Phase I due to PhysioNet/CITI credentialing requirements.
- Justification: Avoids redundant bandwidth usage and ensures 100% reproducible baseline from identical local files.

---

## Decision Record 003: Automated Clinical Validity Evaluation Protocol
- Date: 2026-09-20
- Status: Accepted
- Context: Open Problem 3 identifies that a certified thoracic radiologist panel is not available for real-time human grading during dissertation experiments.
- Decision: Evaluate automated clinical validity through standardized clinical NLP metrics: RadGraph entity/relation agreement (F1 score) and CheXbert diagnostic label concordance.
- Alternatives Considered:
  1. Relying solely on lexical metrics (BLEU, ROUGE, METEOR): Rejected because lexical overlap correlates poorly with clinical correctness (e.g., "no pneumothorax" vs "pneumothorax").
  2. Non-expert manual annotation: Rejected due to lack of radiological credibility.
- Justification: RadGraph and CheXbert are widely accepted in top-tier medical NLP and medical imaging literature (e.g., PhysioNet, Stanford AIMI) as reproducible surrogates for clinical factuality.

---

## Decision Record 004: Patient-Level Data Partitioning and RAG Isolation
- Date: 2026-09-20
- Status: Accepted
- Context: IU X-Ray reports multiple images per patient study (`uid`). Random image-level splitting causes data leakage across train, val, and test partitions, invalidating both generalization metrics and RAG retrieval fairness.
- Decision: Partition data strictly at the patient level (`uid`) using a 70% train / 10% validation / 20% test ratio, stratified on normal vs abnormal diagnosis, using fixed random seed 42. The RAG vector retrieval corpus will be populated strictly from the train partition.
- Alternatives Considered:
  1. Image-level splitting: Rejected because multiple images from the same patient study would leak into test sets.
  2. Global RAG corpus containing all reports: Rejected because retrieving test patient reports creates direct ground truth leakage.
- Justification: Guarantees zero patient leakage, prevents optimistic bias in diagnostic metrics, and enforces strict RAG retrieval integrity.

---

## Decision Record 005: Diagnostic Ground Truth Label Derivation
- Date: 2026-09-20
- Status: Accepted
- Context: Open Problem 1 highlights that IU X-Ray has no native multi-label binary ground truth matrix for standard thoracic pathologies.
- Decision: Extract binary diagnostic labels across 14 standard thoracic categories (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion, Pneumothorax, etc.) using rule-based parsing on Findings and Impression text (CheXpert labeler logic), validated against normalized MeSH and Problems fields.
- Alternatives Considered:
  1. Unsupervised clustering of reports: Rejected due to clinical unpredictability and lack of standard disease mapping.
  2. Relying only on raw MeSH strings: Rejected because MeSH tags in IU X-Ray are inconsistent, mix anatomical descriptors with findings, and lack strict negation tracking.
- Justification: Standard 14-disease labeling aligns IU X-Ray directly with CheXpert and NIH ChestX-ray14 standards, enabling standardized F1, precision, recall, and ROC-AUC evaluation.

---

## Decision Record 006: Localization Ground Truth Proxy for Grad-CAM
- Date: 2026-09-20
- Status: Accepted (Superseded in implementation details by DR016)
- Context: Open Problem 2 notes that IU X-Ray provides no physician-drawn bounding boxes for focal lesions.
- Decision: Evaluate Grad-CAM heatmaps quantitatively using anatomical priors and the Pointing Game protocol (verifying if peak saliency falls within the clinically relevant anatomical zone: cardiomegaly inside cardiac silhouette, pulmonary opacities inside lung fields), supplemented by a curated subset benchmarked against anatomical segmentations.
- Citations:
  - Class Activation Mapping (CAM): Zhou et al., "Learning Deep Features for Discriminative Localization", CVPR 2016.
  - Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
  - Pointing Game Protocol: Zhang et al., "Top-Down Neural Attention by Excitation Backprop", ECCV 2016 / IJCV 2018.
- Alternatives Considered:
  1. Fabricating manual bounding boxes without radiologist oversight: Rejected as non-defensible in a scientific dissertation.
  2. Qualitative visual inspection only: Rejected because MTS requires a continuous quantitative explainability score.
- Justification: Pointing Game accuracy against segmented anatomical compartments provides an objective, literature-backed metric for spatial grounding without unsubstantiated manual annotations.

---

## Decision Record 007: Image View Policy
- Date: 2026-09-20
- Status: Accepted
- Context: IU X-Ray contains both Frontal (PA/AP) and Lateral projections. Some patients have only Frontal, some only Lateral, and most have both.
- Decision: Design Frontal views (PA and AP) as the primary input for single-image VLM diagnostic generation. Record lateral image paths and linkage in metadata to support optional multi-view ablation studies.
- Alternatives Considered:
  1. Frontal-only and discard lateral views completely: Partially rejected; lateral images are kept in metadata to avoid data destruction.
  2. Force mandatory dual-view input: Rejected because patients lacking lateral views would be dropped unnecessarily.
- Justification: Frontal CXR is the clinical standard for initial radiological interpretation and guarantees maximum patient retention without view mismatch artifacts.

---

## Decision Record 008: Target Text Definition and De-identification Token Handling
- Date: 2026-09-20
- Status: Accepted
- Context: IU X-Ray reports contain hospital de-identification tokens (e.g., `XXXX`, `XXXX-year-old`), missing sections, and separate Findings/Impression fields.
- Decision: Standardize target generation text as concatenated Findings and Impression. Normalize de-identification placeholders into semantic tokens (e.g., `[DATE]`, `[AGE]`, `[HOSPITAL]`) where contextual, or strip uninformative repetitions while maintaining sentence syntax. Exclude records with empty Findings and empty Impression from generation tasks.
- Alternatives Considered:
  1. Predicting Impression only: Rejected because Findings contain fine-grained spatial and anatomical observations essential for RAG grounding and Grad-CAM alignment.
  2. Predicting Findings only: Rejected because Impression contains the actionable clinical conclusion.
- Justification: Joint Findings + Impression captures complete radiologist reasoning and aligns with standard clinical report generation benchmarks.

---

## Decision Record 009: Repository Restructuring to Workspace Root
- Date: 2026-09-20
- Status: Accepted
- Context: Nested directory `trustworthy-cxr-vlm` caused Git root confusion with the parent Windows user profile and Git submodule indexing conflicts.
- Decision: Flatten all project directories (`configs/`, `data/`, `docs/`, `reports/`, `src/`, `tests/`) directly to the workspace root `TrustMedicalVLM` mapped directly to GitHub repository `mts-medical-vlm`.
- Alternatives Considered:
  1. Keeping nested `trustworthy-cxr-vlm/` subfolder: Rejected due to redundant directory paths and remote submodule push errors.
- Justification: Standardizes clean repository root access for automated testing, relative config path resolution, and GitHub synchronization.

---

## Decision Record 010: Handling Empty and Partial Report Sections
- Date: 2026-09-20
- Status: Accepted
- Context: Data audit identified 514 reports lacking Findings, 31 lacking Impression, and 25 lacking both sections.
- Decision: For reports with missing Findings but valid Impression, set target text to Impression. For reports with missing Impression but valid Findings, set target text to Findings. Mark reports with both sections empty (23 records after stripping whitespace) as `is_valid_target = False` and assign to split `excluded_empty_target`.
- Alternatives Considered:
  1. Dropping all 514 reports lacking Findings: Rejected because 489 of those reports possess clear diagnostic impressions that are clinically informative.
  2. Imputing missing findings with synthetic text: Rejected to avoid fabricating ground truth observations.
- Justification: Maximizes usable patient cohort (3,666 benchmark patients) while strictly preventing empty target supervision.

---

## Decision Record 011: Multi-Label Pathology Extraction Rules and Negation Filtering
- Date: 2026-09-20
- Status: Accepted
- Context: IU X-Ray lacks native gold multi-label binary indicators for standard thoracic diseases.
- Decision: Implement rule-based clinical regular expressions covering 14 CheXpert/NIH thoracic pathologies, coupled with a 50-character backward-looking negation window and concept corroboration against `Problems` and `MeSH` fields.
- Alternatives Considered:
  1. CheXbert deep learning model inference: Parked for local preprocessing to avoid heavy model downloads prior to user confirmation; the rule-based extractor provides deterministic, reproducible labels.
  2. Unsupervised keyword extraction: Rejected due to failure to capture negation ("no pneumothorax").
- Justification: Ensures accurate, deterministic multi-label assignment across all 3,851 patients with zero GPU overhead.

---

## Decision Record 012: Cryptographic Manifest and File Integrity Tracking
- Date: 2026-09-20
- Status: Accepted
- Context: Medical imaging research requires verifiable reproducibility of raw-to-processed asset linkages.
- Decision: Calculate SHA256 cryptographic hashes for every primary diagnostic image linked to train, val, test, and excluded partitions, saving the structured index in `data/processed/splits_manifest.json`.
- Alternatives Considered:
  1. Plain CSV mapping without hashes: Rejected because silent file modification or corruption cannot be detected.
- Justification: Guarantees strict cryptographic auditability for publication and external validation.

---

## Decision Record 013: Symmetric CheXbert Scoring of Reference and Generated Reports
- Date: 2026-09-20
- Status: Accepted
- Context: The rule-based labeler (Decision Records 005 and 011) extracts ground-truth pathology labels from reference reports. Evaluation of generated reports must use the same labeling procedure to ensure fair comparison: if the labeler's negation detection or vocabulary differs between reference and generated text, metrics become unreliable.
- Decision: Score both reference (ground truth) and VLM-generated reports using CheXbert (Stanford AIMI) as the common diagnostic label extractor. CheXbert produces a 14-class binary pathology vector per report. Clinical accuracy metrics (precision, recall, F1, example-based and label-based) are computed between the two CheXbert label vectors. The existing rule-based labeler is retained for offline preprocessing (label matrix generation, stratified splitting) but not for final evaluation comparisons.
- Alternatives Considered:
  1. Using the rule-based labeler for both reference and generated: Rejected because the rule-based labeler has known false-positive weaknesses on speculative language and limited recall on non-standard phrasing (see labeler validation report, `reports/labeler_validation_100.md`).
  2. Using CheXbert for ground truth but rule-based for generated: Rejected due to asymmetric bias risk — any labeler-specific blind spots would create systematic error in one direction.
  3. Manual radiologist annotation: Not feasible within dissertation timeline.
- Justification: CheXbert is a BERT-based model trained on 187,000+ radiology reports, validated at Stanford AIMI with radiologist-level accuracy. Symmetric application eliminates labeler-induced bias and aligns the evaluation protocol with standard medical report generation benchmarks (e.g., CheXpert competition, RaDialog, CheXagent).

---

## Decision Record 014: Abnormal Rate Reconciliation — Consensus Definition
- Date: 2026-09-20
- Status: Accepted
- Context: The data audit reported 64.19% abnormal rate (2,472 / 3,851) using a simple tag-level definition (MeSH or Problems field is NOT equal to "normal"), while the dataset splits reported 72.24% abnormal rate using the `derive_ground_truth_labels` function which additionally extracted diseases from free-text report sentences. The discrepancy of ~8% was caused by 310 reports where the NLM indexer tagged the study as "normal" but the regex text extractor detected disease-keyword mentions in negated contexts (e.g., "no pleural effusion" → false positive for Effusion).
- Decision: Adopt the NLM curated MeSH/Problems tag as the authoritative normal/abnormal classifier. If MeSH or Problems equals "normal", the study is Normal (is_abnormal=0) regardless of text mentions. Text-based disease extraction is performed only on non-normal studies. This reconciles both the audit and splits to a consistent 64.19% abnormal rate across the full dataset and ~64% across all benchmark splits.
- Alternatives Considered:
  1. Keep the text-based definition (72.2%): Rejected because it overrides radiologist-curated NLM indexation with noisy regex extraction, producing false positives on negated mentions.
  2. Use only tags without text extraction: Rejected because tags lack granularity for the 14-disease multi-label matrix needed for evaluation.
- Justification: The NLM MeSH/Problems tags are assigned by medical librarians following controlled vocabulary standards, making them the most reliable available proxy for overall diagnostic status. Text extraction supplements with fine-grained per-disease labels only where the study is already confirmed abnormal.

---

## Decision Record 015: Uncertainty Quantification via Token Entropy and Sampled CheXbert Consensus
- Date: 2026-09-20
- Status: Accepted
- Context: Medical diagnostic trustworthiness requires quantifying model uncertainty. Prior proposals suggested Monte Carlo (MC) Dropout. However, modern open-source LLM/VLM architectures (e.g., LLaMA, Vicuna, Mistral backbones in LLaVA-Med) do not employ active dropout layers during inference (dropout rate is zero or absent in standard transformer decoders), rendering MC Dropout functionally inactive or inapplicable without architectural modifications.
- Decision: Drop MC Dropout entirely. Implement a two-tiered uncertainty quantification protocol:
  1. **Lexical / Token-Level Uncertainty**: Extract predictive token entropy $H(Y|X) = -\frac{1}{T} \sum_{t=1}^T \sum_{v} P(y_t = v | y_{<t}, X) \log P(y_t = v | y_{<t}, X)$ directly from the autoregressive logit distributions during the greedy decoding pass.
  2. **Semantic / Diagnostic Uncertainty**: Generate 5 stochastic sampled completions per patient ($T=0.7, \text{top\_p}=0.9$) and label each completion using CheXbert. Compute diagnostic confidence as the per-label agreement / consensus across the 5 sampled generations ($\text{Agreement}_c = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(\hat{y}_{c}^{(k)} == \hat{y}_{c}^{(\text{greedy})})$).
- Alternatives Considered:
  1. MC Dropout: Rejected because LLaMA/Mistral architectures contain no dropout during evaluation; enabling training dropout creates out-of-distribution generation artifacts.
  2. Deep Ensembles: Rejected due to prohibitive GPU memory and compute budget requirements.
- Justification: Mean token entropy captures model hesitation at the linguistic token level, while 5-sample CheXbert concordance directly evaluates semantic clinical stability without modifying base model weights.

---

## Decision Record 016: Grad-CAM Redesign via Frozen Vision Encoder Classification Head and Anatomical Region Grounding
- Date: 2026-09-20
- Status: Accepted
- Context: Running Grad-CAM end-to-end through a generative Vision-Language Model requires backpropagating gradients from output token logits through the entire 7B autoregressive LLM back into the vision encoder. This requires caching all intermediate transformer activations in memory during the backward pass (~24+ GB VRAM), making it infeasible on consumer hardware and inefficient even on cloud GPUs. Furthermore, standard Pointing Game protocols that count any saliency peak inside the entire lung fields as a "hit" are trivially satisfied (lungs occupy ~65% of the thoracic image area).
- Decision: Redesign the visual explainability and localization protocol:
  1. **Architecture**: Freeze the pretrained vision encoder (CLIP ViT-L/14 from LLaVA-Med). Train a lightweight multi-label linear/MLP classification head on pooled/patch visual features using the CheXpert 14 disease labels strictly on the **train partition only** (preventing test leakage). Run Grad-CAM backpropagations through this dedicated diagnostic head into the last vision encoder attention/convolutional layer.
  2. **Documented Scientific Limitation**: This setup explains the visual representations learned by the vision encoder, rather than the full multimodal autoregressive decoding reasoning of the LLM.
  3. **Non-Trivial Anatomical Grounding Targets**: Use `torchxrayvision` (PSPNet trained on ChestX-Det) to segment 14 anatomical structures. Evaluate the Pointing Game ($Hit = (x^*, y^*) \in \text{Mask}_{\text{target}}$) and Saliency Mass Ratio (SMR) against disease-specific anatomical compartments rather than global lung fields:
     - *Cardiomegaly*: Heart / cardiac silhouette mask only (hits inside lungs count as misses).
     - *Pleural Effusion*: Bilateral costophrenic angles and Facies Diaphragmatica / lower lung zone mask only.
     - *Pneumothorax*: Apical and lateral peripheral pleural rim mask only.
     - *Mediastinal Widening*: Mediastinum and Aorta mask only.
     - *Fractures*: Clavicle, Scapula, and Spine masks (bony thorax).
- Alternatives Considered:
  1. Full LLM gradient backpropagation: Rejected due to GPU out-of-memory errors and excessive compute requirements.
  2. Whole-lung Pointing Game target: Rejected as clinically uninformative and trivially satisfied.
- Justification: Provides rigorous, reproducible, condition-specific spatial localization metrics with zero test leakage and feasible compute footprint.

---

## Decision Record 017: Diagnostic Evaluation Ontology Standardization
- Date: 2026-09-20
- Status: Accepted
- Context: IU X-Ray data contains disparate labeling schemes: NLM MeSH tags, the rule-based labeler's 14 NIH ChestX-ray14 categories, and CheXbert's 14 observations. Evaluating generated text against reference text requires a unified, clinically standardized evaluation ontology.
- Decision: Standardize strictly on **CheXbert's native 14-observation ontology** (Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices, No Finding) as the primary clinical evaluation ontology for all VLM scoring (S0 through S3). Retain the rule-based NIH-style 14-condition matrix strictly for offline preprocessing, data audit, and stratified dataset partitioning.
- Alternatives Considered:
  1. Using the NIH-style rule-based ontology for evaluation: Rejected because CheXbert's pretrained BERT weights are optimized for the CheXpert 14 observations.
  2. Merging or altering CheXbert label heads: Rejected to preserve full benchmark comparability with external literature.
- Justification: Guarantees exact compatibility with official Stanford AIMI CheXbert scoring protocols and enables direct benchmarking against published medical VLM literature.

