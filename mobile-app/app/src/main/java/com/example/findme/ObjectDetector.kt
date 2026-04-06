package com.example.findme

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Runs yolo26n.onnx on-device via ONNX Runtime to detect sunglasses, keys, or a wallet.
 *
 * Usage:
 *   1. Call [loadModel] once with the application context.
 *   2. Set a [Listener] via [setListener].
 *   3. Set the active target label via [setTarget] ("wallet", "keys", or "glasses").
 *   4. Pass each camera frame to [analyze]; the listener is called with every result.
 *
 * Model I/O (Ultralytics ONNX export with NMS, 3 classes):
 *   Input  — float32 [1, 3, 640, 640], RGB channel-first, values in [0, 1]
 *   Output — float32 [1, 300, 6]  (post-NMS, up to 300 detections)
 *     axis-2 layout: [x1, y1, x2, y2, confidence, class_id]
 *     coordinates are in absolute pixels within the 640×640 input space
 *
 * Detection strategy:
 *   1. Full-frame inference runs on every camera frame.
 *   2. If nothing is found, an async crop cascade fires on a background thread:
 *        – Centre 70% crop  (1.43× effective zoom)
 *        – Top-left  55% crop (1.82× zoom)
 *        – Top-right 55% crop (1.82× zoom)
 *      Only one crop job runs at a time (AtomicBoolean guard).
 */
class ObjectDetector {

    private var listener: DetectorContract.Listener? = null
    private var targetLabel: String = ""

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    companion object {
        private const val TAG = "ObjectDetector"
        private const val INPUT_SIZE = 640
        private const val CONFIDENCE_THRESHOLD = 0.3f
        private const val CENTRE_CROP_FRACTION   = 0.70f
        private const val QUADRANT_CROP_FRACTION = 0.55f

        // Maps voice-command labels to ONNX class indices (from data.yaml)
        private val LABEL_TO_CLASS_INDEX = mapOf(
            "glasses"    to 0,
            "sunglasses" to 0,
            "keys"       to 1,
            "wallet"     to 2
        )
    }

    private val cropExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val isCropRunning = AtomicBoolean(false)

    fun setListener(listener: DetectorContract.Listener) {
        this.listener = listener
    }

    fun setTarget(label: String) {
        targetLabel = label.lowercase()
    }

    /**
     * Load the ONNX model from assets. Call once before any [analyze] calls.
     *
     * @param context       Application or activity context (used to open assets).
     * @param modelFileName File name inside app/src/main/assets/, e.g. "yolo26n.onnx"
     */
    fun loadModel(context: Context, modelFileName: String) {
        try {
            val modelBytes = context.assets.open(modelFileName).readBytes()
            ortEnv = OrtEnvironment.getEnvironment()
            ortSession = ortEnv!!.createSession(modelBytes, OrtSession.SessionOptions())
            Log.i(TAG, "Model loaded: $modelFileName")
            // Log input/output info to help diagnose shape mismatches at runtime.
            ortSession!!.inputNames.forEach { Log.i(TAG, "  input: $it") }
            ortSession!!.outputNames.forEach { Log.i(TAG, "  output: $it") }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: $modelFileName", e)
        }
    }

    /**
     * Analyse one camera frame. Always closes [image] before returning.
     * On full-frame detection, calls [Listener.onResult] immediately.
     * On no detection, fires an async crop cascade (non-blocking).
     */
    fun analyze(image: ImageProxy) {
        val session = ortSession
        val env = ortEnv
        if (session == null || env == null) {
            listener?.onResult(null)
            image.close()
            return
        }

        try {
            val rawBitmap = image.toBitmap()
            image.close()

            val result = inferBitmap(rawBitmap, session, env)

            if (result != null) {
                listener?.onResult(result)
            } else {
                listener?.onResult(null)
                // Fire async crop cascade only when no crop job is already running
                if (isCropRunning.compareAndSet(false, true)) {
                    val bitmapForCrop = rawBitmap
                    cropExecutor.submit {
                        try {
                            listener?.onResult(runCropPasses(bitmapForCrop, session, env))
                        } finally {
                            isCropRunning.set(false)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            listener?.onResult(null)
        }
    }

    fun close() {
        cropExecutor.shutdownNow()
        ortSession?.close()
        ortSession = null
    }

    // ── Crop cascade ──────────────────────────────────────────────────────────

    private fun runCropPasses(src: Bitmap, session: OrtSession, env: OrtEnvironment): DetectorContract.DetectionResult? {
        inferCrop(src, session, env, cxFrac = 0.5f,   cyFrac = 0.5f,   sizeFrac = CENTRE_CROP_FRACTION)
            ?.let { return it }
        inferCrop(src, session, env, cxFrac = 0.275f, cyFrac = 0.275f, sizeFrac = QUADRANT_CROP_FRACTION)
            ?.let { return it }
        return inferCrop(src, session, env, cxFrac = 0.725f, cyFrac = 0.275f, sizeFrac = QUADRANT_CROP_FRACTION)
    }

    /**
     * Crop [src] to a square centred at ([cxFrac], [cyFrac]) with side = [sizeFrac] * min(w,h),
     * run inference, then map coordinates back to full-frame normalised space.
     */
    private fun inferCrop(
        src: Bitmap,
        session: OrtSession,
        env: OrtEnvironment,
        cxFrac: Float,
        cyFrac: Float,
        sizeFrac: Float
    ): DetectorContract.DetectionResult? {
        val w    = src.width
        val h    = src.height
        val side = (minOf(w, h) * sizeFrac).toInt()
        val x0   = (cxFrac * w - side / 2f).toInt().coerceIn(0, w - side)
        val y0   = (cyFrac * h - side / 2f).toInt().coerceIn(0, h - side)
        val crop = Bitmap.createBitmap(src, x0, y0, side, side)

        val result = inferBitmap(crop, session, env) ?: return null

        // Remap crop-normalised coords back to full-frame
        return result.copy(
            normalizedX    = (x0 + result.normalizedX    * side) / w,
            normalizedY    = (y0 + result.normalizedY    * side) / h,
            normalizedArea = result.normalizedArea * (side.toFloat() * side / (w * h))
        )
    }

    // ── Core inference ────────────────────────────────────────────────────────

    /**
     * Letterbox [src] to 640×640, run ONNX inference, return the best detection
     * for the current target class above [CONFIDENCE_THRESHOLD], or null.
     * Coordinates are normalised to [src] dimensions.
     */
    private fun inferBitmap(src: Bitmap, session: OrtSession, env: OrtEnvironment): DetectorContract.DetectionResult? {
        val (letterboxed, scale, padLeft, padTop) = letterbox(src, INPUT_SIZE)
        val contentW = (src.width  * scale).toInt()
        val contentH = (src.height * scale).toInt()

        val floatBuffer = bitmapToFloatBuffer(letterboxed)
        val inputName   = session.inputNames.iterator().next()
        val inputTensor = OnnxTensor.createTensor(
            env,
            floatBuffer,
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )
        val output = session.run(mapOf(inputName to inputTensor))
        val result = parseOutput(output[0].value, padLeft, padTop, contentW, contentH)
        inputTensor.close()
        output.close()
        return result
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /** Convert a 640×640 Bitmap to a CHW float32 buffer normalised to [0, 1]. */
    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val buffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val rChannel = INPUT_SIZE * INPUT_SIZE
        val gChannel = 2 * INPUT_SIZE * INPUT_SIZE

        for (i in pixels.indices) {
            val px = pixels[i]
            buffer.put(i,            ((px shr 16) and 0xFF) / 255f) // R
            buffer.put(i + rChannel, ((px shr 8)  and 0xFF) / 255f) // G
            buffer.put(i + gChannel, ( px         and 0xFF) / 255f) // B
        }
        return buffer
    }

    /**
     * Letterbox-resize [src] to fit within [targetSize]×[targetSize], preserving aspect ratio.
     * Padding (grey 114) is added symmetrically. Matches the letterbox used during training.
     */
    private fun letterbox(src: Bitmap, targetSize: Int): LetterboxResult {
        val scale   = minOf(targetSize.toFloat() / src.width, targetSize.toFloat() / src.height)
        val newW    = (src.width  * scale).toInt()
        val newH    = (src.height * scale).toInt()
        val padLeft = (targetSize - newW) / 2
        val padTop  = (targetSize - newH) / 2

        val scaled = Bitmap.createScaledBitmap(src, newW, newH, true)
        val canvas = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val c      = android.graphics.Canvas(canvas)
        c.drawColor(android.graphics.Color.rgb(114, 114, 114))
        c.drawBitmap(scaled, padLeft.toFloat(), padTop.toFloat(), null)
        scaled.recycle()
        return LetterboxResult(canvas, scale, padLeft, padTop)
    }

    private data class LetterboxResult(
        val bitmap: Bitmap,
        val scale: Float,
        val padLeft: Int,
        val padTop: Int
    )

    /**
     * Parse the raw ONNX output into a [DetectorContract.DetectionResult] for [targetLabel].
     *
     * Expected layout: [1, 300, 6]  (post-NMS)
     *   axis-2 indices: 0=x1, 1=y1, 2=x2, 3=y2, 4=confidence, 5=class_id
     *   Coordinates are absolute pixels in the 640×640 letterboxed input space.
     */
    private fun parseOutput(
        rawOutput: Any,
        padLeft: Int,
        padTop: Int,
        contentW: Int,
        contentH: Int
    ): DetectorContract.DetectionResult? {
        val classIndex = LABEL_TO_CLASS_INDEX[targetLabel] ?: run {
            Log.w(TAG, "Unknown target label: '$targetLabel'")
            return null
        }

        // Shape: float[][][]  →  [1][300][6]
        if (rawOutput !is Array<*>) {
            Log.w(TAG, "Unexpected output type: ${rawOutput::class.java}")
            return null
        }
        @Suppress("UNCHECKED_CAST")
        val detections = (rawOutput as Array<Array<FloatArray>>)[0]  // [300][6]

        var best: DetectorContract.DetectionResult? = null
        var bestConf = CONFIDENCE_THRESHOLD

        for (det in detections) {
            val conf    = det[4]
            val classId = det[5].toInt()
            if (conf > bestConf && classId == classIndex) {
                bestConf = conf
                val x1 = det[0]; val y1 = det[1]
                val x2 = det[2]; val y2 = det[3]
                val normalizedX    = (((x1 + x2) / 2f) - padLeft) / contentW
                val normalizedY    = (((y1 + y2) / 2f) - padTop)  / contentH
                val normalizedArea = ((x2 - x1) * (y2 - y1)) / (contentW * contentH).toFloat()
                best = DetectorContract.DetectionResult(
                    label          = targetLabel,
                    confidence     = conf,
                    normalizedX    = normalizedX.coerceIn(0f, 1f),
                    normalizedY    = normalizedY.coerceIn(0f, 1f),
                    normalizedArea = normalizedArea.coerceIn(0f, 1f)
                )
            }
        }

        return best
    }
}
