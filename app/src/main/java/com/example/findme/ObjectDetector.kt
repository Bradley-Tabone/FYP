package com.example.findme

import androidx.camera.core.ImageProxy

/**
 * Stub for TFLite-based object detection.
 *
 * When the trained model is ready:
 *   1. Copy the .tflite file into app/src/main/assets/
 *   2. Call [loadModel] with the file name (e.g. "findme_model.tflite")
 *   3. Replace the TODO in [analyze] with real inference code
 *
 * Until then, every frame closes the ImageProxy and returns no result so the rest of the
 * pipeline (AudioFeedbackManager, MainActivity) behaves correctly with no model present.
 */
class ObjectDetector {

    data class DetectionResult(
        val label: String,
        val confidence: Float,
        /** Horizontal centre of bounding box: 0.0 = left edge, 1.0 = right edge. */
        val normalizedX: Float,
        /** Vertical centre of bounding box: 0.0 = top edge, 1.0 = bottom edge. */
        val normalizedY: Float,
        /** Bounding box area as a fraction of the full frame area. */
        val normalizedArea: Float
    )

    interface Listener {
        /** Called on the camera executor thread for every analysed frame.
         *  [result] is null when the target object is not detected. */
        fun onResult(result: DetectionResult?)
    }

    private var targetLabel: String = ""
    private var listener: Listener? = null
    private var modelLoaded = false

    fun setListener(listener: Listener) {
        this.listener = listener
    }

    fun setTarget(label: String) {
        targetLabel = label
    }

    /**
     * Load the TFLite model from assets.
     *
     * @param modelFileName  File name inside app/src/main/assets/, e.g. "findme_model.tflite"
     */
    fun loadModel(modelFileName: String) {
        // TODO: initialise the TFLite interpreter
        //
        // val modelBuffer = FileUtil.loadMappedFile(context, modelFileName)
        // val options = Interpreter.Options().apply { numThreads = 2 }
        // interpreter = Interpreter(modelBuffer, options)
        //
        modelLoaded = true
    }

    /**
     * Analyse one camera frame. Must always close [image] before returning.
     * No-op (aside from closing the frame) until a model is loaded via [loadModel].
     */
    fun analyze(image: ImageProxy) {
        if (!modelLoaded) {
            image.close()
            // No result — AudioFeedbackManager will play the "searching" slow beep.
            return
        }

        // TODO: Run inference once model is loaded
        //
        // 1. Convert ImageProxy → Bitmap (or use the YUV planes directly)
        //    val bitmap = image.toBitmap()
        //
        // 2. Resize / normalise to the model's expected input shape
        //
        // 3. Run interpreter
        //    interpreter.run(inputBuffer, outputBuffer)
        //
        // 4. Parse output boxes, filter by targetLabel and confidence threshold
        //
        // 5. Convert the best-match bounding box to normalised coordinates:
        //    val cx = (box.left + box.right)  / 2f / imageWidth
        //    val cy = (box.top  + box.bottom) / 2f / imageHeight
        //    val area = (box.width() * box.height()) / (imageWidth * imageHeight).toFloat()
        //
        // 6. Notify listener
        //    listener?.onResult(DetectionResult(label, confidence, cx, cy, area))
        //    or listener?.onResult(null) if nothing found above threshold

        image.close()
    }
}
