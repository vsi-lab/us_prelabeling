# Day-5 feature-distribution improvement

**Data scope:** TRAIN + VALIDATION ONLY
**Safety status:** TEST SET NOT LOADED OR EVALUATED

No model was retrained. The frozen Day-5 reference populations were reused: Normal=61, Abnormal=194.

## Baseline reproduction

- Maximum centroid-distance replay difference: 2.33e-15
- Maximum centroid-percentile replay difference: 0

## Local kNN results

| Method | Error-detection AUROC | Errors captured in highest-risk 25% |
|---|---:|---:|
| k=3 | 0.682273 | 3/10 |
| k=5 | 0.690000 | 3/10 |
| k=10 | 0.720909 | 6/10 |

Selected development default: **k=10**. k values did not meet the similarity rubric; selected lexicographically by error-detection AUROC, top-25% error capture, then closeness to k=5.

Correct predictions: selected-k percentile mean=63.823, median=73.196.
Incorrect predictions: selected-k percentile mean=87.775, median=93.443.

## Routing ablation

Rules were optimized independently at each accepted count on validation only; envelope masks may be nonnested.

| Strategy | Target coverage | Actual coverage | Errors | Error rate | FN | FP |
|---|---:|---:|---:|---:|---:|---:|
| C | 50% | 50.0% | 0 | 0.00% | 0 | 0 |
| C + Agreement | 50% | 50.0% | 0 | 0.00% | 0 | 0 |
| C + ViewSD | 50% | 50.0% | 0 | 0.00% | 0 | 0 |
| C + Agreement + ViewSD | 50% | 50.0% | 0 | 0.00% | 0 | 0 |
| C + KNN | 50% | 50.0% | 0 | 0.00% | 0 | 0 |
| C + Agreement + KNN | 50% | 50.0% | 0 | 0.00% | 0 | 0 |
| C + ViewSD + KNN | 50% | 50.0% | 0 | 0.00% | 0 | 0 |
| C + Agreement + ViewSD + KNN | 50% | 50.0% | 0 | 0.00% | 0 | 0 |
| C | 60% | 60.0% | 2 | 2.78% | 1 | 1 |
| C + Agreement | 60% | 60.0% | 2 | 2.78% | 1 | 1 |
| C + ViewSD | 60% | 60.0% | 1 | 1.39% | 1 | 0 |
| C + Agreement + ViewSD | 60% | 60.0% | 1 | 1.39% | 1 | 0 |
| C + KNN | 60% | 60.0% | 2 | 2.78% | 1 | 1 |
| C + Agreement + KNN | 60% | 60.0% | 1 | 1.39% | 1 | 0 |
| C + ViewSD + KNN | 60% | 60.0% | 1 | 1.39% | 1 | 0 |
| C + Agreement + ViewSD + KNN | 60% | 60.0% | 1 | 1.39% | 1 | 0 |
| C | 70% | 70.0% | 3 | 3.57% | 2 | 1 |
| C + Agreement | 70% | 70.0% | 2 | 2.38% | 1 | 1 |
| C + ViewSD | 70% | 70.0% | 1 | 1.19% | 1 | 0 |
| C + Agreement + ViewSD | 70% | 70.0% | 1 | 1.19% | 1 | 0 |
| C + KNN | 70% | 70.0% | 2 | 2.38% | 2 | 0 |
| C + Agreement + KNN | 70% | 70.0% | 1 | 1.19% | 1 | 0 |
| C + ViewSD + KNN | 70% | 70.0% | 1 | 1.19% | 1 | 0 |
| C + Agreement + ViewSD + KNN | 70% | 70.0% | 1 | 1.19% | 1 | 0 |
| C | 75% | 75.0% | 3 | 3.33% | 2 | 1 |
| C + Agreement | 75% | 75.0% | 2 | 2.22% | 1 | 1 |
| C + ViewSD | 75% | 75.0% | 2 | 2.22% | 2 | 0 |
| C + Agreement + ViewSD | 75% | 75.0% | 1 | 1.11% | 1 | 0 |
| C + KNN | 75% | 75.0% | 3 | 3.33% | 2 | 1 |
| C + Agreement + KNN | 75% | 75.0% | 1 | 1.11% | 1 | 0 |
| C + ViewSD + KNN | 75% | 75.0% | 2 | 2.22% | 2 | 0 |
| C + Agreement + ViewSD + KNN | 75% | 75.0% | 1 | 1.11% | 1 | 0 |
| C | 80% | 80.0% | 4 | 4.17% | 3 | 1 |
| C + Agreement | 80% | 80.0% | 2 | 2.08% | 1 | 1 |
| C + ViewSD | 80% | 80.0% | 4 | 4.17% | 3 | 1 |
| C + Agreement + ViewSD | 80% | 80.0% | 2 | 2.08% | 1 | 1 |
| C + KNN | 80% | 80.0% | 4 | 4.17% | 3 | 1 |
| C + Agreement + KNN | 80% | 80.0% | 2 | 2.08% | 1 | 1 |
| C + ViewSD + KNN | 80% | 80.0% | 4 | 4.17% | 3 | 1 |
| C + Agreement + ViewSD + KNN | 80% | 80.0% | 2 | 2.08% | 1 | 1 |
| C | 90% | 90.0% | 8 | 7.41% | 5 | 3 |
| C + Agreement | 90% | 90.0% | 6 | 5.56% | 4 | 2 |
| C + ViewSD | 90% | 90.0% | 8 | 7.41% | 5 | 3 |
| C + Agreement + ViewSD | 90% | 90.0% | 6 | 5.56% | 4 | 2 |
| C + KNN | 90% | 90.0% | 8 | 7.41% | 5 | 3 |
| C + Agreement + KNN | 90% | 90.0% | 6 | 5.56% | 4 | 2 |
| C + ViewSD + KNN | 90% | 90.0% | 8 | 7.41% | 5 | 3 |
| C + Agreement + ViewSD + KNN | 90% | 90.0% | 6 | 5.56% | 4 | 2 |

## Conclusion

**B. KNN feature typicality separates validation errors descriptively and can improve simpler confidence-based rules, but adds little beyond the existing view-disagreement signals.**

Strict KNN improvement for at least one simpler route: **True**. Strict improvement after including view SD: **False**. Selected-k unique errors beyond matched confidence/view-SD risk bands: **0**.

This is descriptive validation-only development analysis. It does not establish statistical significance and does not alter the frozen Day-6 operating points.

Optional shrinkage Mahalanobis: Skipped: the numerically stable Ledoit-Wolf fit yielded a degenerate in-sample-calibrated validation percentile distribution; a defensible out-of-fold calibration would add substantial methodology beyond this focused comparison.

Output directory: `F:\VSI - Volume F\Reliability Aware BScan Ultrasound\bscan_prelabeling\outputs\audits\20260817T125159_471804-0700`

**TEST SET NOT LOADED OR EVALUATED**
