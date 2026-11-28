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

---

## Decision Record 018: Train-Only Hybrid Visual–Text Retrieval for S1 (RAG)
- Date: 2026-09-27
- Status: Accepted
- Context: At inference time the only patient inputs are the frontal radiograph and the indication text. IU X-Ray indications are short and often uninformative (e.g. "[AGE] male, chest pain"), so text-only retrieval over reports would retrieve cases that match the indication rather than the image. DR-004 requires the knowledge base to contain train-split reports only.
- Decision: Retrieve the top k = 3 train reports with a hybrid score  s = α · cos(image) + (1 − α) · cos(text).
  1. **Visual channel**: pooled CLS embedding of the frozen LLaVA-Med CLIP ViT-L/14-336 vision tower (the same encoder the VLM sees; DR-016).
  2. **Text channel**: MedCPT query encoder on the patient's indication vs. MedCPT article encoder on each train item's (indication, target report) pair, matching MedCPT's query→article training.
  3. **Index**: exact inner-product search over L2-normalised vectors of the 2,566 train patients only (FAISS IndexFlatIP when available, otherwise an exact NumPy inner product with identical scores; see DR-022). A leakage assertion checks that no val/test uid is present in the index.
  4. **Tuning**: α ∈ {0, 0.25, 0.5, 0.75, 1.0} is selected on VAL only, maximising the mean example-based F1 between the CheXbert labels of the query's reference report and those of each retrieved train report. Test is never used for selection.
  5. **Prompt**: the 3 retrieved reports are placed before the standard query with an explicit instruction that they describe other patients and that only findings visible in the current image should be reported.
- Alternatives Considered:
  1. Text-only retrieval (indication → reports): Rejected because indications carry little diagnostic signal.
  2. Image-only retrieval: Kept as the α = 1 grid point rather than fixed a priori.
  3. BioLinkBERT embeddings: Rejected in favour of MedCPT, which is trained contrastively for biomedical query→document retrieval rather than as a general encoder.
- Justification: Grounds generation in visually similar confirmed cases without leakage, and lets the data (val only) decide the balance between the two channels.

---

## Decision Record 019: Anatomical Target Compartments and Localization Metrics for Grad-CAM
- Date: 2026-09-27
- Status: Accepted
- Context: DR-016 specifies condition-specific targets, but the torchxrayvision ChestX-Det PSPNet segments 14 structures (clavicles, scapulae, lungs, hila, heart, aorta, facies diaphragmatica, mediastinum, weasand, spine) and has no costophrenic-angle, pleural-rim or rib masks. The derived compartments and the metric definitions must be fixed before evaluation.
- Decision:
  1. **Target compartments** (CheXbert observation → mask):
     - Cardiomegaly → Heart.
     - Pleural Effusion → lower third of each lung's vertical extent ∪ Facies Diaphragmatica.
     - Pneumothorax → peripheral lung rim (lung minus lung eroded by 6% of the 512 px frame) ∪ upper quarter of each lung (apex).
     - Enlarged Cardiomediastinum → Mediastinum ∪ Aorta.
     - Fracture → Clavicles ∪ Scapulae ∪ Spine.
  2. **Frame**: heatmaps and masks share one padded-square frame (image padded to square, then resized to 512 px), so no coordinate re-mapping is needed.
  3. **Grad-CAM layer**: layer −2 patch tokens (24 × 24) of the frozen vision tower, i.e. the features LLaVA-Med feeds to its projector.
  4. **Metrics**: Pointing Game hit = arg-max heatmap pixel inside the target mask; Saliency Mass Ratio (SMR) = share of total positive heatmap mass inside the mask. Each is reported with the mask's area fraction as the chance baseline and a Wilson 95% CI for the Pointing Game. An all-zero heatmap counts as a miss with SMR = 0 (it is not dropped).
  5. **Evaluated pairs**: every (patient, condition) pair where the CheXbert reference label is positive and a target exists; reported both over all such pairs and over the subset the head predicts positive.
- Alternatives Considered:
  1. Whole-lung targets: Rejected in DR-016 as trivially satisfied.
  2. Manual bounding boxes: Rejected as unavailable and non-reproducible.
- Limitations: Ribs are not segmented, so rib fractures (the most common kind) fall outside the bony-thorax target and are likely to count as misses. Test prevalence of Pneumothorax is 3 / 734 (CheXbert reference labels), so its per-condition estimate will have a very wide confidence interval.
- Justification: Fixes every free parameter of the localization protocol before any test heatmap is computed.

---

## Decision Record 020: Review Flag Rule and Label-Level Calibration for S3
- Date: 2026-09-27
- Status: Accepted
- Context: DR-015 defines the two uncertainty signals (mean greedy token entropy; per-label CheXbert agreement across 5 samples) but not how they combine into a human-review flag, nor how calibration is measured for stages that output no confidence.
- Decision:
  1. **Combined uncertainty**: each signal is converted to its empirical percentile within the VAL distribution; u = ½ (percentile(entropy) + percentile(1 − consensus)), where consensus is the mean per-label agreement over the 14 observations.
  2. **Flag threshold**: the (1 − budget) quantile of u on VAL with a review budget of 20%, so about one in five reports goes to a radiologist. Test reports are flagged with this fixed threshold.
  3. **Calibration (ECE)**: label-level ECE with 10 equal-width bins over all (report, observation) pairs. S3 confidence in each greedy label = its sample agreement. Stages S0–S2 emit no confidence, so they are scored as asserting every label with confidence 1.0 (the implicit claim of an unhedged report).
  4. **Selective reporting**: clinical F1 is also reported separately for flagged and unflagged test reports, together with the area under the risk–coverage curve (risk = 1 − example-based F1).
- Alternatives Considered:
  1. Fitting a logistic error model on val: Rejected; adds a learned component with only 366 val patients and makes the flag harder to interpret.
  2. Separate thresholds per signal: Rejected; two thresholds on 366 patients overfit more easily than one quantile on a combined score.
- Justification: A fixed review budget is a clinically meaningful and easily explained operating point, and percentile fusion puts two differently scaled signals on a common scale without learned weights.

---

## Decision Record 021: Medical Trustworthiness Score (MTS) Definition and Weights
- Date: 2026-09-27
- Status: Accepted (fixed before any test-split generation or metric was computed)
- Context: The roadmap requires MTS weights to be chosen and justified before test results are seen.
- Decision:
  - Diagnostic D = mean(CheXbert micro-F1 over 14 observations, RadGraph F1 (partial reward)).
  - Reliability R = mean(1 − label-level ECE (DR-020), 1 − hallucination rate), where hallucination rate = generated positive findings (excluding "No Finding") absent from the reference, pooled over the split.
  - Explainability E = mean(Pointing Game hit rate, SMR) (DR-019); E = 0 for S0 and S1, which produce no visual explanation.
  - MTS = 0.4 · D + 0.3 · R + 0.3 · E.
  - Inference latency is reported alongside but excluded from MTS because it depends on hardware and quantization.
  - Sensitivity: MTS is also reported with equal weights (⅓ each) and diagnostic-heavy weights (0.6 / 0.2 / 0.2).
- Justification: Diagnostic correctness is the precondition for any clinical use and receives the largest weight; reliability and explainability are the two trust properties this dissertation adds and receive equal weight. All components lie in [0, 1], so no further normalisation is needed. The sensitivity weights show whether the S0–S3 ranking depends on the weighting.

---

## Decision Record 022: Model Checkpoint Source and Tooling Compatibility
- Date: 2026-09-27
- Status: Accepted
- Context: The official microsoft/llava-med-v1.5-mistral-7b checkpoint requires the original LLaVA code base, which is incompatible with current `transformers` (v5). Two evaluation dependencies also conflict with the local environment.
- Decision:
  1. **VLM checkpoint**: `chaoyinshe/llava-med-v1.5-mistral-7b-hf`, a Hugging Face-format conversion of LLaVA-Med v1.5 (Mistral-7B + CLIP ViT-L/14-336), loaded with `LlavaForConditionalGeneration`. 4-bit NF4 (bitsandbytes) locally with the vision tower and projector kept in FP16; FP16 on Kaggle/Colab for the reported runs. Images are padded to square with the CLIP mean colour, as in LLaVA-1.5.
  2. **RadGraph**: its bundled AllenNLP code needs `transformers < 5`; locally it runs as a subprocess in a separate virtual environment (`RADGRAPH_PYTHON`), and in-process on Kaggle.
  3. **FAISS**: the FAISS DLL is blocked by Windows Application Control on the development machine; the retriever falls back to an exact NumPy inner product, which returns identical scores to IndexFlatIP.
- Limitations: The checkpoint is a community conversion, not an official Microsoft release; this is stated in the dissertation.
- Justification: Keeps the whole pipeline on one maintained library version without changing any metric definition.

---

## Decision Record 023: Correction of the Pleural Effusion Localization Target (amends DR-019)
- Date: 2026-09-30
- Status: Accepted (found on train/val images before any test-split localization was run)
- Context: DR-019 defined the Pleural Effusion target as the lower third of each lung ∪ the PSPNet "Facies Diaphragmatica" mask. Visual overlays and a measurement on 15 train images showed that PSPNet assigns "Facies Diaphragmatica" to the region *below* the lungs (mean rows 330–486 of the 512 px frame vs. a mean lung base at row 387), i.e. the sub-diaphragmatic abdomen, covering 10.5–26.2% of the image (mean 16.9%). Including it raised the mean effusion target from 7.8% to 24.6% of the image, so a heatmap peak anywhere in the upper abdomen would count as an effusion "hit".
- Decision: Pleural Effusion target = lower third of each lung's vertical extent, dilated by the pleural-rim width (6% of the frame, ≈ 30 px) so that it reaches the costophrenic angles and the blunted lung base where fluid collects. The Facies Diaphragmatica mask is no longer used. All other DR-019 targets are unchanged.
- Alternatives Considered:
  1. Keep Facies Diaphragmatica: Rejected; it is abdominal, not pleural, and inflates chance-level hits.
  2. Undilated lower third only: Rejected; fluid blunts the lung edge, so the segmented lung stops above the meniscus and true effusion saliency would fall just outside the mask.
- Justification: Keeps the effusion target anatomically pleural and comparable in size to the other targets, preserving the non-trivial Pointing Game required by DR-016. The chance baseline (mask area fraction) is still reported for every condition.

---

## Decision Record 024: Query Prompt Selection for S0–S3 (val only)
- Date: 2026-10-01
- Status: Accepted (selected on val; no test-split output was scored)
- Context: In the local smoke test the roadmap prompt ("Indication: …. Analyze this chest radiograph and provide detailed findings and diagnostic impression.") produced patient-education prose rather than radiology-style FINDINGS/IMPRESSION text. Prompt wording changes clinical content, so it was selected on the first 40 val patients (by uid), with LLaVA-Med in 4-bit, greedy decoding, and CheXbert scoring against the val reference labels.
- Round 1 (criterion fixed in advance: CheXbert micro-F1 over 14 observations): `report_v2` (explicit FINDINGS/IMPRESSION with an anatomical checklist and "state normal findings when the study is normal") scored highest (micro-F1 0.351 vs 0.159 for the roadmap prompt). However, it called 36 of 40 studies normal (18 are normal) and found 0 of the 37 abnormal reference labels: its micro-F1 came almost entirely from the frequent "No Finding" label. Micro-F1 therefore rewards a near-constant "normal" output on this cohort.
- Round 2: two further templates without the normality cue were added (`report_v4`, `report_v5`), and all five were compared on the same 40 val patients by **CheXbert macro-F1 (14)**, which weights every observation equally and so rewards detecting abnormalities. This change of criterion was made after seeing the round-1 val result and is reported as such. Results (macro-F1 / micro-F1 / correct-of-predicted abnormal labels / studies called normal / FINDINGS+IMPRESSION format rate):
  - `baseline_v1` (roadmap prompt): 0.111 / 0.159 / 8 of 64 / 7 / 0.00
  - `report_v2`: 0.045 / 0.351 / 0 of 6 / 36 / 0.97
  - `report_v3`: 0.071 / 0.160 / 5 of 37 / 8 / 0.03
  - `report_v4`: 0.079 / 0.234 / 2 of 17 / 22 / 0.95
  - `report_v5`: 0.095 / 0.161 / 5 of 45 / 12 / 0.12
- Decision: Use the roadmap prompt `baseline_v1` for all stages (S0–S3), built from the cleaned indication by `src/models/prompts.py::build_query` (trailing full stops are no longer doubled). Raw generations and scores: `results/prompt_selection/`.
- Observed trade-off: explicit report formatting makes LLaVA-Med conservative (few findings, mostly "normal"), while free-form prompts elicit more true and many more false abnormal findings. The roadmap prompt detects the most abnormalities but also has the highest false-positive count; this is exactly what the hallucination rate and the S1 retrieval context are meant to expose and reduce.
- Limitations: 40 val patients is a small sample and the macro-F1 differences are modest. The selected prompt does not produce FINDINGS/IMPRESSION formatting, which lowers lexical scores (BLEU/ROUGE-L) against formatted reference reports; CheXbert and RadGraph score clinical content and are unaffected by formatting.
- Justification: Macro-F1 is the criterion aligned with the clinical purpose (detecting abnormalities) on a cohort where 38% of reports are "No Finding"; the selected prompt is also the one pre-specified in the roadmap, so S0 remains the planned baseline.

---

## Decision Record 025: RAG Prompt Layout for S1–S3 (val only)
- Date: 2026-10-01
- Status: Accepted (selected on val; no test-split output was scored)
- Context: In the first S1 smoke run (DR-018 layout: header, then the 3 retrieved reports, then the query last), LLaVA-Med returned an empty report for 7 of 20 patients: it emitted a space and then the end-of-sequence token. Forcing a minimum length (`min_new_tokens`) was tried and rejected: the model then produced only newline tokens up to the length limit, so the reports stayed empty. The likely cause is that, with the retrieved reports immediately before the end of the user turn, the model treats the task as already answered.
- Decision: Use the **query_first** layout for S1–S3: the standard query (DR-024), then the reference-report header and the 3 retrieved train reports, then a closing instruction: "Now write the findings and impression for THIS patient's chest radiograph, describing only what is visible in this image." (`src/models/prompts.py::rag_prompt`, `rag.layout` in the config).
- Evidence (first 40 val patients by uid, 4-bit, greedy; empty / CheXbert macro-F1 / micro-F1 / RadGraph F1 / hallucination rate / ECE):
  - S0 roadmap prompt, no retrieval (reference): 0 / 0.111 / 0.159 / 0.066 / 0.875 / 0.189
  - context_first (original layout): 6 of 40 / 0.117 / 0.309 / 0.221 / 0.778 / 0.120
  - query_first (selected): 0 of 40 / 0.099 / 0.337 / 0.273 / 0.818 / 0.113
- Selection rule (stated after the val result and disclosed as such): a layout must produce a non-empty report for every patient; context_first fails this for 15% of val patients, which is a validity failure rather than a metric trade-off. query_first is also higher on micro-F1, RadGraph F1 and ECE. It is lower on macro-F1 (0.099 vs 0.117), where the gap amounts to one or two rare-label hits at n = 40, and has a higher hallucination rate (0.818 vs 0.778).
- Observations to report: (1) both layouts show the model copying phrasing from retrieved reports, including de-identification tokens such as "[REDACTED]"; query_first shows this more (lower mean token entropy, 0.41 vs 0.75 on the first 20 patients, and higher BLEU/ROUGE-L). (2) These are val tuning numbers and not results; the S0–S3 comparison is made only on the test split.
- Raw generations: `results/s1/generations_val_ragctx40.jsonl`, `results/s1/generations_val_ragqry40.jsonl`; scores: `results/s1/rag_layout_val/`.
- Addendum (2026-10-02, measured with `src/evaluation/nlg_metrics.py::copy_overlap` on the same 40 val patients): mean share of a report's word 4-grams found verbatim in its closest retrieved report — S0 (never shown the retrieved reports; chance baseline) 0.002, context_first 0.237, query_first 0.601; reports with more than half of their 4-grams copied — 0, 7 and 21 of 40. The query_first layout therefore avoids empty reports largely by copying retrieved text, and its higher val micro-F1 and RadGraph F1 may reflect that copying. Copy overlap is now reported for every RAG stage so this can be assessed on test.

---

## Decision Record 026: Chance-Corrected Explainability in the MTS (amends DR-021)
- Date: 2026-10-02
- Status: Accepted (decided by the author before any test-split generation or metric was computed)
- Context: DR-021 set Explainability E = mean(Pointing Game, SMR), with E = 0 for stages without a visual explanation. Val localization (DR-023) is at chance level overall (Pointing Game 0.153, SMR 0.110, mean target area 0.109), yet under DR-021 such heatmaps still add about 0.3 x 0.13 to the MTS, so the score would reward producing a heatmap rather than a useful one.
- Decision: E = mean(cc(Pointing Game), cc(SMR)), with cc(x) = max(0, (x - c) / (1 - c)) and c the mean target-mask area fraction over the evaluated (patient, condition) pairs, i.e. the score of a uniform heatmap. cc is 0 at or below chance and 1 for perfect localization. E = 0 is kept for S0 and S1. Weights (0.4 / 0.3 / 0.3) and the D and R components are unchanged. The chance level is reported in the ablation table (`localization_chance`).
- Alternatives Considered:
  1. Keep DR-021: Rejected; chance-level saliency would raise S2 and S3 MTS without any localization ability.
  2. Per-condition correction weighted by pair counts: Rejected for simplicity; conditions with very few pairs (e.g. 3 Pneumothorax test cases) would carry unstable chance estimates.
- Justification: A kappa-style correction is a standard way to remove chance agreement, keeps E in [0, 1], and makes the explainability axis measure localization above chance, which is what DR-016 intends to evaluate.

---

## Decision Record 027: Final Scoring Runs on Kaggle Alongside Generation
- Date: 2026-11-26
- Status: Accepted (no test-split output exists yet)
- Context: Windows Smart App Control on the development laptop now blocks PyTorch's DLLs (`torch_python.dll`, Code Integrity event 3077, seen since 2026-11-05). Generation was already planned for Kaggle FP16 (DR-001), but CheXbert scoring, RadGraph and Grad-CAM localization also need torch and can no longer run locally. Turning Smart App Control off is irreversible without a Windows reset and was not done.
- Decision: `notebooks/kaggle_s0_s3_generation.ipynb` now generates and scores in one Kaggle session: S0/S1 generation (FP16), the four `evaluate_stage.py` runs (review flag fitted on val, applied to test), CLIP feature extraction, test localization, `build_ablation.py` and `bootstrap_ci.py`, packaged as `phase2_results.zip`. RadGraph runs in its own venv (transformers < 5, DR-022). The Grad-CAM head trained locally on the train split is uploaded as an input and never retrained. CLIP features are re-extracted on Kaggle, so val localization is recomputed with them and printed next to the committed val numbers as a consistency check. Scoring dependencies are installed before generation so an install failure surfaces immediately, and the full pytest suite runs on Kaggle (informational).
- Alternatives Considered:
  1. 4-bit generation and scoring locally: blocked by Smart App Control.
  2. WSL2 with CUDA: not installed; needs admin rights, a reboot and an unverified environment.
- Limitations: Kaggle-extracted CLIP features may differ numerically (GPU, FP16) from the local features used for the head's val selection; the val re-check quantifies this.
- Justification: Keeps the FP16 final-run plan, leaves the laptop's security settings alone, and produces every metric from one recorded commit.

---

## Decision Record 028: Phase III Improvements After the v1 Test Run; Review Flag v2
- Date: 2026-11-27
- Status: Accepted. The v1 S0–S3 test results (commit 0e09b75) are kept unchanged as the pre-registered result; everything below is a disclosed post-hoc improvement, selected on val only and reported separately as v2.
- Context: The v1 test run showed that the DR-020 review flag does not rank errors: test AURC 0.673 against a full-coverage risk of 0.662 (risk = 1 - example F1). On val its 5-fold CV AURC is 0.665 against 0.671; mean greedy token entropy alone is close to uninformative (0.647).
- Decision (review flag v2): `src/pipeline/tune_review_flag.py` compares reference-free per-report risk scorers on val by 5-fold cross-validated AURC (seed 42): the DR-020 percentile combination, each of nine single features (sign chosen on the training fold), and ridge regression on all nine (`src/uncertainty/review_features.py`). Lowest CV AURC is selected, refit on all val, and the threshold set at the val 80th percentile (20% budget). It is applied to test once.
- Val evidence (CV AURC): retrieved-report label disagreement 0.557 (selected), retrieved-majority mismatch 0.571, ridge (l2=100) 0.572, sample disagreement 0.573, DR-020 combination 0.665, token entropy 0.647.
- Test (applied once, S1 v1 generations): AURC 0.629 vs full-coverage risk 0.662 (DR-020 flag: 0.673); flag rate 14.9% (the feature is discrete, so ties put fewer reports above the val threshold); example F1 0.248 flagged vs 0.353 unflagged (DR-020: 0.313 vs 0.346).
- Note: the selected signal measures disagreement among the retrieved train reports, i.e. retrieval uncertainty rather than the generator's own. S3's calibrated confidences (label agreement, DR-015) are unchanged; only the review flag is replaced.
- Outputs: `results/s3/review_flag_v2_selection.json`, `review_flagger_v2.json`, `review_flag_v2_{val,test}.json`.

---

## Decision Record 029: Grad-CAM Target Layer = Head Hidden Tokens (Phase III, val-selected)
- Date: 2026-11-27
- Status: Accepted (selected on val; applied to test once; v1 test localization in `results/s2/localization_test*` is kept)
- Context: v1 Grad-CAM (DR-016) targets the frozen encoder's layer -2 patch tokens. Its val Pointing Game (0.153, 124 pairs) did not exceed chance (0.109). Raw CLIP ViT tokens contain a few very large outlier channels, which can dominate the gradient-weighted channel sum regardless of the class.
- Decision: Target the head's per-token hidden activations (after LayerNorm-Linear-GELU, before mean-pooling) — `xai.cam_layer: "hidden"`, `src/xai/gradcam.py`. Because mean-pooling is followed by a linear classifier, Grad-CAM at this layer equals the class activation map (CAM) of Zhou et al.; it is still a gradient-based explanation of the same frozen-encoder head, with no retraining.
- Val evidence (366 patients, 124 pairs; Pointing Game [Wilson 95% CI] / chance): tokens 0.153 [0.10, 0.23] / 0.109; hidden 0.250 [0.18, 0.33] / 0.109. Per condition, hidden: Cardiomegaly 0.182 (chance 0.078), Pleural Effusion 0.895 (0.197), Enlarged Cardiomediastinum 0.036 (0.043), Fracture 0.150 (0.196), Pneumothorax 0/2.
- Test (applied once, 207 pairs): Pointing Game 0.203 [0.15, 0.26] vs chance 0.099 (v1: 0.155); SMR 0.104 (chance 0.099); Cardiomegaly 0.115 (v1 0.038, chance 0.076); Pleural Effusion 0.846 (v1 0.731); Enlarged Cardiomediastinum 0.000; Fracture 0.381; Pneumothorax 0/3.
- Outputs: `results/s2/localization_{val,test}_camhidden{.csv,_summary.json}`.

---

## Decision Record 030: LoRA Fine-Tuning of LLaVA-Med on Train Reports (Phase III, v2)
- Date: 2026-11-28
- Status: Accepted (design fixed before any v2 output exists; v1 results in `results/` stay unchanged)
- Context: The v1 test run (zero-shot LLaVA-Med) reached CheXbert micro-F1 0.154 (S0) and 0.309 (S1), macro-F1 0.115/0.089, with S1 copying on average 56% of its 4-grams from retrieved reports and calling 490 of 734 test studies normal (281 in the references). LLaVA-Med was never trained to write IU X-Ray reports.
- Decision: Train one LoRA adapter (r 16, alpha 32, dropout 0.05; q/k/v/o/gate/up/down of the language model; vision tower and projector frozen) on the 2,566 TRAIN patients for one epoch (lr 2e-4, cosine, 3% warm-up, gradient accumulation 8, FP16 on Kaggle). Each patient is one example: with probability 0.5 the S1 prompt with its top-3 retrieved TRAIN reports of OTHER patients (same hybrid index and alpha as S1, self excluded; `results/finetune/retrieval_train.csv`), otherwise the S0 prompt. Target = the patient's own reference report (truncated at 200 tokens); loss on report tokens only. The adapter with the lowest loss on the first 60 val patients (checked every 80 optimizer steps) is kept. The same adapter serves S0 and S1; decoding, prompts, sampling, scoring, review budget and MTS weights are unchanged.
- Why mix S0 and S1 prompts: one adapter covers both stages, and since the target never equals a retrieved report, the model is trained to use retrieved reports as evidence rather than to transcribe them.
- Selection rule: val loss only; no generation-based tuning on val and nothing on test. Phase III test results are reported as v2 next to v1 in `results_v2/` (via `MTS_RESULTS_DIR`), with S2 using the DR-029 localization and the S3 review flag re-selected on v2 val outputs by DR-028's procedure.
- Alternatives Considered: (1) QLoRA locally: the 6 GB GPU and a 5 GB Windows page file cannot load the 7B model for training. (2) A CXR-specific base model (CheXagent, LLaVA-Rad): rejected, it would replace the DR-016 encoder link and the dissertation's base model. (3) Training S0-only prompts: rejected, S1 prompts would be out of distribution for the adapter.
- Run: `notebooks/kaggle_v2_finetune.ipynb` (fine-tune, generate with `--adapter`, score; resumable across sessions); output `phase3_results.zip`.

## Decision Record 031: LoRA Adapter Rejected (Template Collapse); Undefined Metrics Reported as Missing
- Date: 2026-11-28
- Status: Accepted (v1, i.e. zero-shot LLaVA-Med with DR-028 flag and DR-029 localization, stays the reported pipeline; v2 kept in `results_v2/` as a negative result)
- Context: The DR-030 Kaggle run (2x T4, FP16) trained 321 optimizer steps; val loss fell from 1.760 to 0.661 (best = last step). Generations show template collapse. Val (S1, 366 patients): 41 distinct reports, 80.1% identical, CheXbert macro-F1 0.040 vs 0.126 for v1, micro-F1 over 5 conditions 0.015 vs 0.103, hallucination rate 0.929 vs 0.806. Test agrees: S0 728/734 reports are the same normal template and assert no positive finding at all (hallucination rate undefined); S1 569/734 identical, macro-F1 0.045 vs 0.089, copy overlap 0.355 vs 0.557. The higher test micro-F1 (S0 0.364, S1 0.334) comes from predicting "No Finding" for almost everyone, not from better findings.
- Decision: (1) Do not adopt the adapter; the rejection rests on the val outputs above (val loss alone did not detect the collapse). (2) When a stage asserts no positive finding, its hallucination rate, reliability and MTS are reported as missing ("—"), never filled in; paired-bootstrap deltas on such a metric are reported as undefined. `build_ablation.py` and `bootstrap_ci.py` implement this; other missing metrics still raise.
- Why: Token-level cross-entropy on IU X-Ray is dominated by the frequent normal templates, so the lowest-loss adapter is the one that best reproduces them. A model that writes one report for every patient is not more trustworthy, whatever its micro-F1.
- Alternatives Considered: (1) Report v2 as the improved pipeline on its micro-F1: rejected as misleading. (2) Retrain with abnormal oversampling or generation-based val selection: possible future work; it needs another full Kaggle run and is not done here. (3) Treat the undefined hallucination rate as 0: rejected, it would fabricate a perfect reliability term.
- Outputs: `results_v2/` (finetune/train_log.json, S0/S1 generations and metrics, S3 flag re-selected by DR-028 on v2 val: ridge, test AURC 0.574 vs full-coverage risk 0.654, ablation and bootstrap tables). Adapter weights are not committed.

## Decision Record 032: Balanced LoRA Fine-Tuning with Generation-Based Checkpoint Selection (Phase III, v3)
- Date: 2026-11-28
- Status: Accepted (recipe fixed before any v3 output exists; v1 stays the reported pipeline unless v3 wins on val)
- Context: The DR-030 adapter collapsed to a normal template although val loss kept falling (DR-031). Only 38.5% of train reports are CheXbert "No Finding", so the collapse is not explained by class balance alone: with a frozen projector the language model leans on its text prior, and token cross-entropy rewards the safest average report.
- Decision: Config block `finetune_v3`, run with `finetune_vlm.py --recipe v3`. (1) Rare-finding oversampling: each epoch draws train examples with weight sqrt(n_max / n_rarest positive CheXbert reference label), clipped to [1, 5] (normal reports weight 1); the share of sampled reports with a finding rises from 40% to 64%. (2) The multi-modal projector is trained in full alongside LoRA (peft modules_to_save). (3) lr 1e-4, 2 epochs (642 optimizer steps). (4) Checkpoint selection on generations: at step 0 (zero-shot, adapter disabled) and every 107 steps, greedy S1 reports for a fixed random subset of 100 val patients are labelled by CheXbert; score = micro-F1 over the 13 findings (No Finding excluded), and a candidate with fewer than 50% distinct reports is ineligible. The best candidate is kept; if zero-shot wins, no adapter is used and v1 remains the result.
- Why: Each change targets one cause of the v2 collapse, and the selection rule measures the failure directly on val rather than through loss. Letting zero-shot compete means v3 cannot replace v1 unless it is better on val.
- Alternatives Considered: (1) Selecting by macro-F1 over 14 labels: too noisy on 100 patients for rare labels. (2) Unlikelihood or contrastive loss against templates: more complex, not standard for LLaVA fine-tuning. (3) Full fine-tuning: does not fit 2x T4.
- Run: `notebooks/kaggle_v3_finetune.ipynb` (outputs `results_v3/`, `phase3_v3_results.zip`; packaging no longer depends on scoring succeeding).

## Decision Record 033: Non-VLM Reference Baselines (Constant Normal Report, Copy of Top-1 Retrieved Report)
- Date: 2026-11-28
- Status: Accepted (evaluation-only controls; no model or threshold is changed)
- Context: S1 copies 56% of its 4-grams from retrieved reports, and the v2 adapter that wrote one normal template for everyone reached a higher micro-F1 than S1 (DR-031). Without controls, the S0-S3 table cannot say how much comes from the VLM.
- Decision: `src/pipeline/run_baselines.py` writes two untuned, train-only baselines in the generations format, scored by the unchanged `evaluate_stage.py`, and `src/evaluation/compare_baselines.py` reports paired bootstrap deltas (2,000 resamples, seed 42). (1) `template`: the most frequent TRAIN report whose CheXbert reference label is "No Finding" only, given to every patient. (2) `retrieval_top1`: the TRAIN report of the top-1 S1 neighbour, with no VLM.
- Disclosure: the first template rule was "the most frequent TRAIN report" overall. On test, CheXbert labels that report ("The heart and lungs have [REDACTED] in the interval...") as Cardiomegaly, so it is not a normal-report control; the rule was restricted to No-Finding-only reports after seeing this. The original is kept as `template_any` (test micro-F1 0.135, macro-F1 0.018).
- Test results: template micro-F1 0.364 (above every VLM stage), macro-F1 0.040, RadGraph 0.261, hallucination undefined (no finding asserted). retrieval_top1 micro-F1 0.280, macro-F1 0.121, RadGraph 0.274, hallucination 0.823, BLEU-4 0.112, ROUGE-L 0.314. S1 minus retrieval_top1: micro-F1 +0.029 [0.002, 0.056], RadGraph +0.017 [0.008, 0.026], macro-F1 -0.032 [-0.055, -0.008], hallucination +0.025 [-0.016, 0.064].
- RadGraph locally: Windows Smart App Control blocks the radgraph venv's torch DLL, so RadGraph ran with the main interpreter plus the venv's radgraph/transformers 4.57.6 packages (USE_TF=0). It reproduces the Kaggle v1 S1 test RadGraph F1 exactly (0.290730).

## Decision Record 034: External Localization Test on NIH ChestX-ray14 Radiologist Boxes
- Date: 2026-11-28
- Status: Accepted (protocol fixed before any NIH result; a 10-image pipeline smoke test was run and discarded)
- Context: S2 is scored against anatomical compartments on IU X-Ray only. Two limitations follow: the targets are not drawn lesions, and there is no second hospital.
- Decision: `src/pipeline/eval_external_localization.py` scores the unchanged IU-trained S2 head (train split only) on all 984 radiologist-drawn boxes of NIH ChestX-ray14 `BBox_List_2017.csv` (880 images, CC0, Kaggle nih-chest-xrays/data). Label mapping fixed in code before evaluation: Cardiomegaly->Cardiomegaly, Effusion->Pleural Effusion, Atelectasis->Atelectasis, Pneumothorax->Pneumothorax, Pneumonia->Pneumonia, Infiltrate->Lung Opacity, Mass->Lung Lesion, Nodule->Lung Lesion. Metrics as in S2 (Pointing Game with Wilson CI, SMR, chance = box area fraction) for both CAM layers (patch tokens, pre-specified; hidden, DR-029), plus a content-free centre-Gaussian prior (sigma = 1/4 side) as an anatomy-prior baseline. Boxes are mapped into the same padded-square 512 px frame as the heatmaps.
- Data handling: images stay git-ignored (`data/external/*`, path `paths.nih_bbox_dir`). Per-file Kaggle API downloads were throttled after ~100 images, so the run uses `notebooks/kaggle_nih_external.ipynb` (CPU, data mounted).
- Alternatives Considered: VinDr-CXR (radiologist boxes, but credentialed access); ChestX-Det (segmentations, NIH subset, less standard).

## Decision Record 035: MAIRA-2 as a Second, Chest-Specific Base VLM
- Date: 2026-11-28
- Status: Accepted (comparison model; LLaVA-Med remains the pipeline's base model)
- Context: LLaVA-Med S0/S1 are weak, and copying the top retrieved report nearly matches S1 (DR-033). The user asked for stronger models. DR-030 had rejected replacing the base model because the S2 head and S1 image retrieval use LLaVA-Med's CLIP encoder; adding a second model as a comparison keeps that design intact.
- Decision: MAIRA-2 (microsoft/maira-2, 7B, research licence, gated) runs as S0 on the same patients with the same inputs (frontal image + indication; no lateral, prior, technique or comparison) and greedy decoding (max 256 new tokens), plus 5 samples (T=0.7, top-p 0.9) for the S3 uncertainty layer, scored by the unchanged evaluation code. No retrieval: MAIRA-2's prior-study slot is for the same patient, so other patients' reports would be mislabelled as priors. The review flag v2 procedure (DR-028) is re-run on MAIRA-2 val outputs (`tune_review_flag.py --stage s0`; copy feature = 0).
- Contamination check: the MAIRA-2 paper (arXiv 2406.04449) and model card state that IU X-Ray was not used for training and serve as an external test set; one third-party paper (arXiv 2512.00271) lists IU-Xray among its training data, which contradicts the authors. We follow the authors and report the discrepancy.
- Caveats: MAIRA-2 writes only the FINDINGS section while references join findings and impression; S2 Grad-CAM explains LLaVA-Med's encoder, not MAIRA-2, so explainability is not credited to MAIRA-2 in the MTS.
- Alternatives Considered: CheXagent (training data may include IU X-Ray / OpenI; not confirmed), LLaVA-Rad (sources not fully listed, custom code).
- Run: `notebooks/kaggle_maira2.ipynb` (transformers 4.51.3, HF_TOKEN Kaggle secret, 3-patient smoke test first) -> `results_maira2/`.
