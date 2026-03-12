---
name: au12-math
description: AU12 facial action unit intensity computation from MediaPipe landmarks. Use when working on packages/ml_core/au12.py, face_mesh.py, or any code computing facial metrics, landmark extraction, IOD normalization, or FACS scoring.
---

# AU12 Mathematical Specification (§7)

## Landmark indices (§7.1–7.2)

Mouth corners: landmarks[61] (left), landmarks[291] (right).
Left eye: landmarks[33] (outer), landmarks[133] (inner).
Right eye: landmarks[362] (inner), landmarks[263] (outer).

## IOD computation (§7.3)

Eye centers are midpoints: `e_l = mean([L33, L133], axis=0)`, `e_r = mean([L362, L263], axis=0)`.
IOD = 3D Euclidean distance between e_l and e_r.

## AU12 scoring (§7.4)

```
d_mouth = norm(L291 - L61)            # 3D Euclidean
raw_ratio = d_mouth / iod
i_au12 = raw_ratio - b_neutral        # baseline-subtracted
score_au12 = max(0, alpha * i_au12)    # alpha default 5.0
```

Output MUST be clamped to [0.0, 5.0]. Epsilon guard (1e-6) on IOD to prevent division by zero.

## Calibration

During calibration phase (`is_calibrating=True`): accumulate raw_ratio into buffer, set `b_neutral = mean(buffer)`, return 0.0. Inference requires calibrated baseline or raises `ValueError`.

## v2.0 corrections

Landmark indexing uses `landmarks[i]` not `landmarks.landmark[i]`. Full type annotations required. Output hard-clamped to 5.0.
