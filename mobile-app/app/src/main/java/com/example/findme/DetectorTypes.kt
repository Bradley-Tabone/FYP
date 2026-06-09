package com.example.findme

object DetectorTypes {

    data class DetectionResult(
        /** type of detected object */
        val label: String, 
        /** confidence of detection */
        val confidence: Float,
        /** normalized values for X and Y that show where the object is in frame horizontally and vertically respectively */
        val normalizedX: Float,
        val normalizedY: Float,
        /** Bounding box area as a fraction of the full frame area */
        val normalizedArea: Float,
        /** Normalized Box width as a fraction of frame width */
        val normalizedW: Float = 0f,
        /** Normalized Box height as a fraction of frame height */
        val normalizedH: Float = 0f,
        /** Source frame width and height in pixels depending on angle the device is held. Used to compute preview crop offset. */
        val sourceW: Int = 0,
        val sourceH: Int = 0
    )

    interface Listener {
        /** Receives either a real detection, or null if nothing was found */
        fun onResult(result: DetectionResult?) 

        /** Function used to determine the brightness in frame which is then used to decide if torch must be on or off */
        fun onLuma(luma: Int) {}
    }
}
