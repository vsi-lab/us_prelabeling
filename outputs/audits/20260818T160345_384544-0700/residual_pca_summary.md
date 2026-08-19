# Residual-PCA feature typicality — validation-only follow-up

Data scope: **TRAIN + VALIDATION ONLY**  
Test status: **TEST SET NOT LOADED OR EVALUATED**

## Method

The frozen Day-4 classifier direction was removed from class-centered 512-D pooled features. Separate 32-component whitened PCAs were fit using all correctly classified training eyes in each true class (121 Normal, 388 Abnormal). Typicality is the mean Euclidean distance to 10 same-class training references. Training calibration distances exclude the query eye itself; validation uses the Day-4 predicted class.

## Score comparison

| Method | Spearman with confidence | Error-detection AUROC |
|---|---:|---:|
| Centroid cosine | -0.704157 | 0.747273 |
| Existing kNN (k=10) | -0.559803 | 0.720909 |
| Residual PCA (32-D whitened, k=10) | 0.343197 | 0.430000 |

## Frozen balanced-rule check

The one error accepted by the frozen balanced validation rule had residual-PCA percentile **1.652893**. It **was not** caught by the prespecified `>95` defer gate.

| Rule | Coverage | Accepted errors | Accepted FN | Accepted FP |
|---|---:|---:|---:|---:|
| Frozen balanced rule | 75.00% | 1 | 1 | 0 |
| Frozen balanced rule + defer ResidualPCA percentile >95 | 75.00% | 1 | 1 | 0 |

This is a fixed validation-only follow-up. No cutoff was searched, no model was retrained, and the frozen routing artifact was not modified.

TEST SET NOT LOADED OR EVALUATED.
