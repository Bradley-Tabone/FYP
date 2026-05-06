package com.example.findme

/**
 * Shared result type and listener interface used by [ObjectDetector].
 * MainActivity, BoundingBoxOverlay, and AudioFeedbackManager all reference these types.
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
        val normalizedArea: Float,
        /** Box width as a fraction of frame width (0.0–1.0). */
        val normalizedW: Float = 0f,
        /** Box height as a fraction of frame height (0.0–1.0). */
        val normalizedH: Float = 0f,
        /** Source frame width in pixels (after rotation). Used to compute preview crop offset. */
        val sourceW: Int = 0,
        /** Source frame height in pixels (after rotation). Used to compute preview crop offset. */
        val sourceH: Int = 0
    )

    interface Listener {
        fun onResult(result: DetectionResult?)
    }
}
