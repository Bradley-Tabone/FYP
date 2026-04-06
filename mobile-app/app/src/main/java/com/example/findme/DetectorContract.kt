package com.example.findme

/**
 * Shared result type and listener interface used by both [ObjectDetector] (YOLO)
 * and [ObjectDetectorRTDETR] (RT-DETR). MainActivity, BoundingBoxOverlay, and
 * AudioFeedbackManager all reference these types, so switching detectors only
 * requires changing the two lines in MainActivity marked "Model switch".
 */
object DetectorContract {

    data class DetectionResult(
        val label: String,
        val confidence: Float,
        /** Horizontal centre: 0.0 = left edge, 1.0 = right edge. */
        val normalizedX: Float,
        /** Vertical centre: 0.0 = top edge, 1.0 = bottom edge. */
        val normalizedY: Float,
        /** Bounding box area as a fraction of the full frame area. */
        val normalizedArea: Float
    )

    interface Listener {
        fun onResult(result: DetectionResult?)
    }
}
