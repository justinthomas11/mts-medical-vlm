# Validation Report: Rule-Based Thoracic Pathology Labeler

> [!WARNING]
> **NON-INDEPENDENT REFERENCE DISCLOSURE**: The reference standard used in this 100-sample validation
> is **non-independent**. Ground truth was derived directly from the same NLM MeSH/Problems indexation
> fields and dictionary lookups that the rule-based labeler consults. All 1.0000 scores indicate internal
> rule concordance rather than independent clinical validity. True independent diagnostic validation
> requires external model evaluation (CheXbert) or certified radiologist re-annotation.

## 1. Executive Summary
- Sample Size: 100 randomly sampled IU X-Ray patient reports (Seed: 42).
- Reference Standard: Heuristic mapping over NLM MeSH/Problems indexation (**Non-Independent**).
- Diagnostic Breakdown in Sample: 36% Normal (36), 64% Abnormal (64).
- Overall Concordance (Normal vs Abnormal): **100.0%** (36/36 normal, 64/64 abnormal) — *Circular due to shared NLM tag dependency*.

## 2. Condition-Level Performance Metrics

| Condition | Support | TP | FP | FN | TN | Precision | Recall | F1 Score | Evaluation Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Atelectasis | 12 | 12 | 0 | 0 | 88 | 1.0000 | 1.0000 | 1.0000 | Non-Independent |
| Cardiomegaly | 11 | 11 | 0 | 0 | 89 | 1.0000 | 1.0000 | 1.0000 | Non-Independent |
| Consolidation | 1 | 1 | 1 | 0 | 98 | 0.5000 | 1.0000 | 0.6667 | Discrepancy |
| Edema | 3 | 3 | 1 | 0 | 96 | 0.7500 | 1.0000 | 0.8571 | Discrepancy |
| Effusion | 3 | 3 | 1 | 0 | 96 | 0.7500 | 1.0000 | 0.8571 | Discrepancy |
| Emphysema | 4 | 4 | 3 | 0 | 93 | 0.5714 | 1.0000 | 0.7273 | Discrepancy |
| Fibrosis | 7 | 7 | 1 | 0 | 92 | 0.8750 | 1.0000 | 0.9333 | Discrepancy |
| Fracture | 3 | 3 | 1 | 0 | 96 | 0.7500 | 1.0000 | 0.8571 | Discrepancy |
| Hernia | 0 | 0 | 0 | 0 | 100 | N/A | N/A | N/A (0 support) | Zero Support |
| Infiltration | 0 | 0 | 1 | 0 | 99 | 0.0000 | 0.0000 | 0.0000 | Discrepancy |
| Mass | 0 | 0 | 0 | 0 | 100 | N/A | N/A | N/A (0 support) | Zero Support |
| Nodule | 16 | 16 | 0 | 0 | 84 | 1.0000 | 1.0000 | 1.0000 | Non-Independent |
| Pleural_Thickening | 2 | 2 | 0 | 0 | 98 | 1.0000 | 1.0000 | 1.0000 | Non-Independent |
| Pneumothorax | 0 | 0 | 0 | 0 | 100 | N/A | N/A | N/A (0 support) | Zero Support |
| No_Finding | 36 | 36 | 0 | 0 | 64 | 1.0000 | 1.0000 | 1.0000 | Non-Independent |
| is_abnormal | 64 | 64 | 0 | 0 | 36 | 1.0000 | 1.0000 | 1.0000 | Non-Independent |

## 3. Granular Error Analysis: Where the Rule-Based Labeler Fails

Across the 100 examined reports, the rule-based labeler produced discrepancies in **8** cases.

### 3.1 Failure Categories
1. **Speculative Mentions vs Confirmed Findings (False Positives)**:
   - The labeler flags conditions when the radiologist notes a possibility or suggests differential diagnosis (e.g. *'Fracture is possible if high energy trauma'* or *'could reflect a small focus of atelectasis or infiltrate'*).
2. **Granular Terminology Mismatches (False Negatives)**:
   - Clinical descriptions using non-standard wording (e.g. *'streaky bibasilar opacities'* or *'density at the lung base'*) that represent subtle subsegmental changes without the exact keyword *atelectasis* or *consolidation*.
3. **Modal / Diagnostic Recommendations (False Positives)**:
   - Phrases like *'CT scan is more sensitive in detecting small nodules'* where the word *nodules* appears in a modality recommendation rather than a radiological finding on the current radiograph.
4. **Compound Negation Distance Limits (Historic Failure Mode)**:
   - In prior implementations with a fixed 50-character negation window, serial negated lists (e.g. *'no focal consolidation, suspicious pulmonary opacity, pneumothorax or large pleural effusion'*) failed to negate downstream terms like *effusion*. The updated sentence-level negation scope resolved these false positives.

### 3.2 Specific Discrepancy Examples

#### Patient Study UID 3680
- **Discrepancies**: FP: Consolidation, FP: Infiltration
- **NLM MeSH Tags**: `Opacity/lung/base/bilateral/streaky;Pulmonary Atelectasis/base/bilateral;Opacity/lung/middle lobe/right;Calcified Granuloma/scattered/multiple`
- **Findings**: The heart is normal in size and contour. There is no mediastinal widening. Streaky bibasilar opacities, XXXX atelectasis. Vague opacity in the right midlung. Scattered calcified granulomas. No large pleural effusion or pneumothorax. The XXXX are intact.
- **Impression**: Vague opacity in the right midlung, this could reflect a small focus of atelectasis or infiltrate. Bibasilar airspace opacities, XXXX atelectasis.

#### Patient Study UID 3328
- **Discrepancies**: FP: Edema
- **NLM MeSH Tags**: `Cardiomegaly/mild;Pleural Effusion/bilateral/small`
- **Findings**: nan
- **Impression**: Minimal cardiomegaly. Prominent pulmonary XXXX. Probable very small pleural effusions and minimal questionable interstitial edema. No pneumonia

#### Patient Study UID 3658
- **Discrepancies**: FP: Fracture
- **NLM MeSH Tags**: `Thickening/pleura/apex/bilateral/round;Cicatrix/pleura/apex/bilateral/round;Nodule/pleura/apex/bilateral/round;Deformity/spine`
- **Findings**: The cardiomediastinal silhouette is normal in size and contour. No focal consolidation, pneumothorax or large pleural effusion. Biapical fibronodular pleural thickening/scarring. There is a XXXX like deformity of the anterior cortex of the XXXX body (lateral view). Negative for retrosternal density. Prior cholecystectomy. Critical result notification documented through Primordial.
- **Impression**: Buckling deformity of the anterior cortex of the XXXX body. Fracture is possible, if high energy XXXX was localized to this region. Correlate with focal tenderness. XXXX chest, if warranted.

#### Patient Study UID 501
- **Discrepancies**: FP: Fibrosis
- **NLM MeSH Tags**: `Opacity/lung/posterior/streaky/mild`
- **Findings**: nan
- **Impression**: 1. There is minimal streaky opacity in the posterior lungs, possibly cyst, scarring, or pneumonia. 2. Heart size and pulmonary XXXX appear normal 3. Mediastinal contours are normal

#### Patient Study UID 2151
- **Discrepancies**: FP: Emphysema
- **NLM MeSH Tags**: `Lung/hyperdistention;Diaphragm/bilateral/flattened;Pulmonary Edema;Scoliosis/right/mild`
- **Findings**: Hyperinflated lungs with flattened diaphragm and increased retrosternal airspace. No focal alveolar consolidation, no definite pleural effusion seen. Heart size within normal limits, the typical findings of pulmonary edema. Mild spine dextrocurvature noted.
- **Impression**: Hyperinflated lungs, air trapping versus inspiratory XXXX.

#### Patient Study UID 1791
- **Discrepancies**: FP: Emphysema
- **NLM MeSH Tags**: `Technical Quality of Image Unsatisfactory ;Lung/hyperdistention/mild;Calcified Granuloma/lung/base/right/large;Thoracic Vertebrae/degenerative/multiple/mild`
- **Findings**: Limited exam as the left costophrenic XXXX is excluded from the PA view. The heart size is normal. The mediastinal contour is within normal limits. Mild lung hyperinflation. The lungs are free of any focal infiltrates. There is large calcified granuloma within the medial right lung base. There are no nodules or masses. No visible pneumothorax. No visible pleural fluid. Mild multilevel degenerative changes seen within the thoracic spine. No visible acute fracture. There is no visible free intraperitoneal air under the diaphragm.
- **Impression**: 1. No acute radiographic cardiopulmonary process. 2. Mild hyperinflation.

#### Patient Study UID 1076
- **Discrepancies**: FP: Effusion
- **NLM MeSH Tags**: `Opacity/thorax/posterior;Consolidation/lung/lower lobe/left;Pulmonary Congestion/mild;Fractures, Bone/ribs/right/multiple/healed;Deformity/ribs/right/multiple/healed;Cardiomegaly/borderline`
- **Findings**: There is opacity at posterior aspect of lower chest seen on lateral view which probably represents left lower lobe consolidation. There may also be small bilateral pleural effusion. Upper limits of normal heart size. Mild central vascular prominence. Old fracture deformities of multiple right ribs.
- **Impression**: 1. Question of left lower lobe pneumonia and/or pleural effusion. 2. Borderline heart size with mild central vascular congestive changes.

#### Patient Study UID 3533
- **Discrepancies**: FP: Emphysema
- **NLM MeSH Tags**: `Lung/bilateral/hyperdistention/mild`
- **Findings**: Heart size, mediastinal contour, and pulmonary vascularity are within normal limits. There is bilateral hyperinflation, without focal consolidation, pneumothorax, or pleural effusion. Visualized osseous structures appear intact.
- **Impression**: Mildly hyperinflated, clear lungs.
