# Day-7 one-time held-out test evaluation

No model was retrained. No threshold or routing rule was searched or tuned on test data. Original unmasked source images and the locked Day-3/Day-4 deterministic preprocessing were used.

## Test cohort

- Eyes: 120
- Subjects: 95
- Normal: 32
- Abnormal: 88

## Classification

| Model | AUROC | Balanced accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Day3PerImageMAX | 0.977628 | 0.944602 | 0.920455 | 0.968750 | 0.952941 |
| Day4MultiViewFeatureMAX | 0.960227 | 0.875000 | 0.875000 | 0.875000 | 0.911243 |

## Frozen selective routing

Accepted error rate uses accepted eyes—not the full cohort—as its denominator.

| Rule | Accepted | Deferred | Coverage | Review | Accepted errors | Accepted error rate | Accepted FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conservative zero-error rule | 70 | 50 | 58.33% | 41.67% | 1 | 1.43% | 0 |
| Balanced agreement and view-SD rule | 97 | 23 | 80.83% | 19.17% | 5 | 5.15% | 4 |

## Subject-level bootstrap confidence intervals

| Analysis | Metric | Point estimate | 95% CI | Valid replicates |
|---|---|---:|---:|---:|
| Day4Classification | AUROC | 0.960227 | [0.924246, 0.987952] | 2000/2000 |
| Day4Classification | BalancedAccuracy | 0.875000 | [0.805073, 0.935004] | 2000/2000 |
| Day4Classification | Sensitivity | 0.875000 | [0.800000, 0.939759] | 2000/2000 |
| Day4Classification | Specificity | 0.875000 | [0.757386, 0.973684] | 2000/2000 |
| balanced_agreement_view_sd | AcceptedErrorRate | 0.051546 | [0.010635, 0.097826] | 2000/2000 |
| balanced_agreement_view_sd | AcceptedSensitivity | 0.945946 | [0.888889, 0.987504] | 2000/2000 |
| balanced_agreement_view_sd | AcceptedSpecificity | 0.956522 | [0.857143, 1.000000] | 2000/2000 |
| balanced_agreement_view_sd | Coverage | 0.808333 | [0.731692, 0.876055] | 2000/2000 |
| conservative_zero_error | AcceptedErrorRate | 0.014286 | [0.000000, 0.046875] | 2000/2000 |
| conservative_zero_error | AcceptedSensitivity | 1.000000 | [1.000000, 1.000000] | 2000/2000 |
| conservative_zero_error | AcceptedSpecificity | 0.909091 | [0.692308, 1.000000] | 2000/2000 |
| conservative_zero_error | Coverage | 0.583333 | [0.487995, 0.674797] | 2000/2000 |

Risk/coverage figure: `outputs\figures\day7_heldout_test\20260817T164200_422134-0700\day7_test_risk_coverage.png`

HELD-OUT TEST EVALUATION COMPLETE.
NO TEST-BASED THRESHOLD TUNING WAS PERFORMED.
FROZEN VALIDATION ROUTING RULES WERE APPLIED WITHOUT MODIFICATION.
