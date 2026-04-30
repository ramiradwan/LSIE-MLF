---
name: au12-math
description: AU12 facial action unit intensity computation from MediaPipe landmarks. Use when working on packages/ml_core/au12.py, face_mesh.py, or any code computing facial metrics, landmark extraction, IOD normalization, or FACS scoring.
---

# AU12 Mathematical Specification (§7A)

## Landmark indices (§7A.1–7A.2)

Mouth corners: landmarks[61] (left), landmarks[291] (right).
Left eye: landmarks[33] (outer), landmarks[133] (inner).
Right eye: landmarks[362] (inner), landmarks[263] (outer).

## IOD computation (§7A.3)

Eye centers are midpoints: `e_l = mean([L33, L133], axis=0)`, `e_r = mean([L362, L263], axis=0)`.
IOD = 3D Euclidean distance between e_l and e_r.

## AU12 scoring (§7A.4)

```
d_mouth = norm(L291 - L61)            # 3D Euclidean
raw_ratio = d_mouth / iod
i_au12 = raw_ratio - b_neutral        # baseline-subtracted
score_au12_bounded = tanh(alpha * max(0, i_au12))  # alpha default 6.0
```

Output MUST be bounded to [0.0, 1.0] via tanh soft-saturation. Epsilon guard (1e-6) on IOD to prevent division by zero.

## Calibration

During calibration phase (`is_calibrating=True`): accumulate raw_ratio into buffer, set `b_neutral = mean(buffer)`, return 0.0. Inference requires calibrated baseline or raises `ValueError`.

## Implementation corrections

Landmark indexing uses `landmarks[i]` not `landmarks.landmark[i]`. Full type annotations required. Output uses tanh soft-saturation with default `alpha=6.0` and remains bounded to [0.0, 1.0].
