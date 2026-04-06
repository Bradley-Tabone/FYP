package com.example.findme

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * RF-DETR Base variant of the object detector.
 *
 * Drop-in replacement for [ObjectDetector] / [ObjectDetectorRTDETR].
 * To switch in MainActivity, change the two lines marked "Model switch":
 *   ObjectDetectorRFDETR()  +  loadModel(this, "rfdetr_base.onnx")
 *
 * Model I/O (RF-DETR Base ONNX export, opset 17, 3 classes):
 *   Input  — float32 [1, 3, 560, 560], RGB channel-first
 *            ImageNet-normalised: (pixel/255 - mean) / std
 *            mean=[0.485, 0.456, 0.406]  std=[0.229, 0.224, 0.225]
 *   Output — TWO tensors (unlike YOLO/RT-DETR which have one):
 *            dets   float32 [1, 300, 4]  — (cx, cy, w, h) normalised [0,1]
 *            labels float32 [1, 300, 4]  — logits: [Sunglasses, Keys, Wallet, no-object]
 *
 * Pre-processing uses direct square resize (no letterbox padding), so the
 * normalised box coordinates map 1-to-1 onto the original frame.
 *
 * No crop cascade — RF-DETR is too heavy on CPU for extra passes.
 *
 * Performance notes vs YOLO26n:
 *   - 107 MB ONNX vs 9.4 MB  — much slower to load and infer on CPU
 *   - NNAPI intentionally disabled: DINOv2 attention ops are unsupported by NNAPI
 *     and cause silent session-creation failure (no detections); CPU-only instead
 */
class ObjectDetectorRFDETR {

    private var listener: DetectorContract.Listener? = null
    private var targetLabel: String = ""

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // Temporal smoothing: require this many consecutive detections before reporting.
    // Filters out single-frame false positives (camera panning over a similar-looking object)
    // without significantly delaying genuine detections (a real object stays in frame).
    @Volatile private var consecutiveHits = 0
    private val LOCK_ON_FRAMES = 2

    companion object {
        private const val TAG = "ObjectDetectorRFDETR"
        private const val INPUT_SIZE = 560
        private const val CONFIDENCE_THRESHOLD = 0.80f
        private const val NO_OBJECT_CLASS = 3  // last logit is DETR's no-object class

        // ImageNet normalisation constants
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

        private val LABEL_TO_CLASS_INDEX = mapOf(
            "glasses"    to 0,
            "sunglasses" to 0,
            "keys"       to 1,
            "wallet"     to 2
        )
    }

    fun setListener(listener: DetectorContract.Listener) { this.listener = listener }
    fun setTarget(label: String) { targetLabel = label.lowercase(); consecutiveHits = 0 }

    fun loadModel(context: Context, modelFileName: String) {
        try {
            Log.i(TAG, "Loading model: $modelFileName …")
            val modelBytes = context.assets.open(modelFileName).readBytes()
            Log.i(TAG, "  Model bytes read: ${modelBytes.size}")
            ortEnv = OrtEnvironment.getEnvironment()
            // RF-DETR uses a DINOv2 transformer backbone whose attention ops are
            // not supported by NNAPI; skip NNAPI to avoid silent session-creation failure.
            ortSession = ortEnv!!.createSession(modelBytes, OrtSession.SessionOptions())
            Log.i(TAG, "Model loaded OK: $modelFileName")
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
            Log.w(TAG, "analyze(): model not loaded (session=$session env=$env) — skipping frame")
            listener?.onResult(null)
            image.close()
            return
        }
        try {
            val rawBitmap = image.toBitmap()
            image.close()
            val raw = inferBitmap(rawBitmap, session, env)
            if (raw != null) consecutiveHits++ else consecutiveHits = 0
            // Only report once we've seen the detection in LOCK_ON_FRAMES consecutive frames.
            listener?.onResult(if (consecutiveHits >= LOCK_ON_FRAMES) raw else null)
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            consecutiveHits = 0
            listener?.onResult(null)
        }
    }

    fun close() {
        ortSession?.close()
        ortSession = null
    }

    // ── Core inference ────────────────────────────────────────────────────────

    private fun inferBitmap(
        src: Bitmap, session: OrtSession, env: OrtEnvironment
    ): DetectorContract.DetectionResult? {
        // RF-DETR uses direct square resize — no letterbox, no padding
        val resized     = Bitmap.createScaledBitmap(src, INPUT_SIZE, INPUT_SIZE, true)
        val floatBuffer = bitmapToFloatBuffer(resized)
        resized.recycle()

        val inputName   = session.inputNames.iterator().next()
        val inputTensor = OnnxTensor.createTensor(
            env, floatBuffer,
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )
        val outputs = session.run(mapOf(inputName to inputTensor))
        val result  = parseOutputs(outputs[0].value, outputs[1].value)
        inputTensor.close()
        outputs.close()
        return result
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Convert a 560×560 Bitmap to a CHW float32 buffer with ImageNet normalisation:
     *   channel_value = (pixel/255 - mean) / std
     */
    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val pixels   = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val buffer   = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val rChannel = INPUT_SIZE * INPUT_SIZE
        val gChannel = 2 * INPUT_SIZE * INPUT_SIZE
        for (i in pixels.indices) {
            val px = pixels[i]
            buffer.put(i,            (((px shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0])
            buffer.put(i + rChannel, (((px shr 8)  and 0xFF) / 255f - MEAN[1]) / STD[1])
            buffer.put(i + gChannel, (( px         and 0xFF) / 255f - MEAN[2]) / STD[2])
        }
        return buffer
    }

    /**
     * Parse RF-DETR's two output tensors.
     *
     * dets   [1, 300, 4] — (cx, cy, w, h) normalised [0,1], directly in frame space
     * labels [1, 300, 4] — logits: [Sunglasses, Keys, Wallet, no-object]
     *
     * A detection is accepted when:
     *   1. softmax(logits)[targetClass] > CONFIDENCE_THRESHOLD
     *   2. targetClass has the highest score across ALL four logits
     *      (including no-object — ensures we reject low-confidence hits)
     */
    private var parseLogCounter = 0

    private fun parseOutputs(rawDets: Any, rawLabels: Any): DetectorContract.DetectionResult? {
        val classIndex = LABEL_TO_CLASS_INDEX[targetLabel] ?: run {
            Log.w(TAG, "Unknown target label: '$targetLabel'")
            return null
        }
        if (rawDets !is Array<*> || rawLabels !is Array<*>) {
            Log.w(TAG, "Unexpected output types: dets=${rawDets::class.java} labels=${rawLabels::class.java}")
            return null
        }
        @Suppress("UNCHECKED_CAST")
        val dets   = (rawDets   as Array<Array<FloatArray>>)[0]  // [300][4]
        @Suppress("UNCHECKED_CAST")
        val labels = (rawLabels as Array<Array<FloatArray>>)[0]  // [300][4]

        // Log raw output values every ~30 frames to help diagnose zero-detection issues
        if (parseLogCounter % 30 == 0) {
            val sampleProbs = softmax(labels[0])
            Log.d(TAG, "parseOutputs: target='$targetLabel'(idx=$classIndex) dets[0]=${dets[0].toList()} logits[0]=${labels[0].toList()} probs[0]=${sampleProbs.toList()}")
        }
        parseLogCounter++

        var best: DetectorContract.DetectionResult? = null
        var bestConf = CONFIDENCE_THRESHOLD

        for (i in dets.indices) {
            val logits = labels[i]        // [4] raw logits
            val probs  = softmax(logits)  // [4] probabilities
            val conf   = probs[classIndex]

            if (conf <= bestConf) continue
            // Reject if any other class (including no-object) scored higher
            if (probs.indices.any { it != classIndex && probs[it] >= conf }) continue

            bestConf = conf
            val cx = dets[i][0]; val cy = dets[i][1]
            val w  = dets[i][2]; val h  = dets[i][3]
            // Coordinates already normalised [0,1] in original frame space
            best = DetectorContract.DetectionResult(
                label          = targetLabel,
                confidence     = conf,
                normalizedX    = cx.coerceIn(0f, 1f),
                normalizedY    = cy.coerceIn(0f, 1f),
                normalizedArea = (w * h).coerceIn(0f, 1f)
            )
        }
        if (parseLogCounter % 30 == 1 && best != null) {
            Log.d(TAG, "parseOutputs: best=${best.label} conf=${best.confidence} x=${best.normalizedX} y=${best.normalizedY}")
        }
        return best
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max  = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum  = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }
}
