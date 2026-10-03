# CEFM Calibration Update

The image calibration gate now accepts two controlled reference types:

1. A standardized 20 mm x 20 mm square marker.
2. A straight ruler with clearly visible 1 mm graduations.

The system prefers the square marker when detected. If it is not detected, it searches for a ruler with a long baseline and regular 1 mm tick pattern. A calibration result is accepted only when the detected scale has sufficient regularity and confidence.

## Patient capture protocol

- Place the complete calibration reference beside the lesion.
- Keep the reference and lesion on the same skin surface/plane.
- Keep the camera approximately parallel to the skin.
- Make the complete lesion and complete reference visible.
- Use even lighting and avoid glare, strong shadows and blur.
- Do not use filters or excessive digital zoom.
- For ruler mode, use a ruler with clearly visible 1 mm graduations.

If neither reference can be validated, the image is rejected and no physical diameter is reported.

## Validation

The included test suite covers the existing image-quality gate plus a synthetic 1 mm ruler calibration case.
