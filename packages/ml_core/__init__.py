# packages/ml_core — §4.D Multimodal ML Processing
# Shared mathematical and inference utilities.
#
# Planned addition: likes_detection.py
#   Lightweight OpenCV subroutine for counting screen-tap "heart bubble"
#   animations from the scrcpy video feed (Module A frames already in memory
#   for MediaPipe). Uses pixel-variance frame differencing over a static ROI
#   bounding box in the bottom-right corner of the screen. Returns an integer
#   tap count per segment, appended to the InferenceHandoffPayload alongside
#   AU12 and acoustic data. No additional model or GPU load required.
