# Preliminary Per-Class Results (13 of 18 runs complete)
# Generated: $(date)

## Per-class mean ± std across 3 seeds (test set AP50)

| Variant | AP50_MH | AP50_PH | AP50_WLPH |
|---|---|---|---|
| baseline | 0.858 ± 0.009 | 0.570 ± 0.018 | 0.751 ± 0.016 |
| cbam | 0.850 ± 0.006 | 0.587 ± 0.011 | 0.735 ± 0.016 |
| p2 | 0.858 ± 0.001 | 0.579 ± 0.005 | 0.740 ± 0.015 |
| sizeaware | 0.860 ± 0.005 | 0.589 ± 0.004 | 0.730 ± 0.007 |

## Key findings (preliminary)

1. MH (manholes): all variants ~ceiling, no meaningful difference
2. PH (potholes): CBAM and sizeaware show consistent +0.017 to +0.019 improvement
3. WLPH (waterlogged): all enhancements show -0.011 to -0.021 regression
4. Aggregate mAP50 unchanged because gains on PH offset losses on WLPH

## Status
- baseline, cbam, p2, sizeaware: 3/3 seeds complete
- combined: 1/3 seeds (currently training seed 1337)
- distill: not yet started
