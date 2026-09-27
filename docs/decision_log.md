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
