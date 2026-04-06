package com.example.findme

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import java.nio.FloatBuffer

/**
 * RT-DETR-l variant of the object detector.
 *
 * Drop-in replacement for [ObjectDetector]. Identical usage — swap the class name in
 * MainActivity and load "rtdetr_l.onnx" instead of "yolo26n.onnx".
 *
 * Model I/O (Ultralytics RT-DETR ONNX export, opset 17, imgsz=320, 3 classes):
 *   Input  — float32 [1, 3, 320, 320], RGB channel-first, values in [0, 1]
 *   Output — float32 [1, 300, 7]  (post-NMS, up to 300 detections)
 *     axis-2 layout: [x1, y1, x2, y2, score_Sunglasses, score_Keys, score_Wallet]
 *     Coordinates: normalised [0, 1] in the 320×320 letterboxed input space.
 *
 * Performance choices vs ObjectDetector (YOLO):
 *   - INPUT_SIZE 320 instead of 640 — 4× less computation per inference
 *   - No crop cascade — RT-DETR is too slow on CPU to afford 3 extra passes per miss
 *   - NNAPI requested at session creation — offloads to device GPU/DSP where available
 *
 * To switch back to YOLO in MainActivity:
 *   - Replace ObjectDetectorRTDETR() with ObjectDetector()
 *   - Change loadModel(this, "rtdetr_l.onnx") to loadModel(this, "yolo26n.onnx")
 */
class ObjectDetectorRTDETR {

    private var listener: DetectorContract.Listener? = null
    private var targetLabel: String = ""

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    companion object {
        private const val TAG = "ObjectDetectorRTDETR"
        private const val INPUT_SIZE = 320
        private const val CONFIDENCE_THRESHOLD = 0.3f

        private val LABEL_TO_CLASS_INDEX = mapOf(
            "glasses"    to 0,
            "sunglasses" to 0,
            "keys"       to 1,
            "wallet"     to 2
        )
    }

    fun setListener(listener: DetectorContract.Listener) { this.listener = listener }
    fun setTarget(label: String) { targetLabel = label.lowercase() }

    fun loadModel(context: Context, modelFileName: String) {
        try {
            val modelBytes = context.assets.open(modelFileName).readBytes()
            ortEnv = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            try {
                options.addNnapi()
                Log.i(TAG, "NNAPI execution provider enabled")
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI not available, falling back to CPU: ${e.message}")
            }
            ortSession = ortEnv!!.createSession(modelBytes, options)
            Log.i(TAG, "Model loaded: $modelFileName")
            ortSession!!.inputNames.forEach  { Log.i(TAG, "  input:  $it") }
            ortSession!!.outputNames.forEach { Log.i(TAG, "  output: $it") }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: $modelFileName", e)
        }
    }

    fun analyze(image: ImageProxy) {
        val session = ortSession
        val env     = ortEnv
        if (session == null || env == null) {
            listener?.onResult(null)
            image.close()
            return
        }
        try {
            val rawBitmap = image.toBitmap()
            image.close()
            listener?.onResult(inferBitmap(rawBitmap, session, env))
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            listener?.onResult(null)
        }
    }

    fun close() {
        ortSession?.close()
        ortSession = null
    }

    // ── Core inference ────────────────────────────────────────────────────────

    private fun inferBitmap(src: Bitmap, session: OrtSession, env: OrtEnvironment): DetectorContract.DetectionResult? {
        val (letterboxed, scale, padLeft, padTop) = letterbox(src, INPUT_SIZE)
        val contentW = (src.width  * scale).toInt()
        val contentH = (src.height * scale).toInt()

        val floatBuffer = bitmapToFloatBuffer(letterboxed)
        val inputName   = session.inputNames.iterator().next()
        val inputTensor = OnnxTensor.createTensor(
            env, floatBuffer,
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )
        val output = session.run(mapOf(inputName to inputTensor))
        val result = parseOutput(output[0].value, padLeft, padTop, contentW, contentH)
        inputTensor.close()
        output.close()
        return result
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val buffer   = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val rChannel = INPUT_SIZE * INPUT_SIZE
        val gChannel = 2 * INPUT_SIZE * INPUT_SIZE
        for (i in pixels.indices) {
            val px = pixels[i]
            buffer.put(i,            ((px shr 16) and 0xFF) / 255f)
            buffer.put(i + rChannel, ((px shr 8)  and 0xFF) / 255f)
            buffer.put(i + gChannel, ( px         and 0xFF) / 255f)
        }
        return buffer
    }

    private fun letterbox(src: Bitmap, targetSize: Int): LetterboxResult {
        val scale   = minOf(targetSize.toFloat() / src.width, targetSize.toFloat() / src.height)
        val newW    = (src.width  * scale).toInt()
        val newH    = (src.height * scale).toInt()
        val padLeft = (targetSize - newW) / 2
        val padTop  = (targetSize - newH) / 2
        val scaled  = Bitmap.createScaledBitmap(src, newW, newH, true)
        val canvas  = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val c       = android.graphics.Canvas(canvas)
        c.drawColor(android.graphics.Color.rgb(114, 114, 114))
        c.drawBitmap(scaled, padLeft.toFloat(), padTop.toFloat(), null)
        scaled.recycle()
        return LetterboxResult(canvas, scale, padLeft, padTop)
    }

    private data class LetterboxResult(
        val bitmap: Bitmap, val scale: Float, val padLeft: Int, val padTop: Int
    )

    /**
     * RT-DETR output layout: [1, 300, 7]
     *   0=x1, 1=y1, 2=x2, 3=y2  — normalised [0,1] in the 320×320 letterboxed space
     *   4=score_Sunglasses, 5=score_Keys, 6=score_Wallet
     *
     * A detection is accepted only when the target class has the highest score AND
     * that score exceeds CONFIDENCE_THRESHOLD.
     */
    private fun parseOutput(
        rawOutput: Any, padLeft: Int, padTop: Int, contentW: Int, contentH: Int
    ): DetectorContract.DetectionResult? {
        val classIndex = LABEL_TO_CLASS_INDEX[targetLabel] ?: run {
            Log.w(TAG, "Unknown target label: '$targetLabel'")
            return null
        }
        if (rawOutput !is Array<*>) {
            Log.w(TAG, "Unexpected output type: ${rawOutput::class.java}")
            return null
        }
        @Suppress("UNCHECKED_CAST")
        val detections = (rawOutput as Array<Array<FloatArray>>)[0]  // [300][7]

        var best: DetectorContract.DetectionResult? = null
        var bestConf = CONFIDENCE_THRESHOLD

        for (det in detections) {
            val conf = det[4 + classIndex]
            if (conf <= bestConf) continue
            // Reject if another class scored higher
            val maxScore = maxOf(det[4], det[5], det[6])
            if (conf < maxScore) continue

            bestConf = conf
            // Coords are normalised [0,1] → scale to pixels in the letterboxed space
            val x1 = det[0] * INPUT_SIZE;  val y1 = det[1] * INPUT_SIZE
            val x2 = det[2] * INPUT_SIZE;  val y2 = det[3] * INPUT_SIZE
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
        return best
    }
}
