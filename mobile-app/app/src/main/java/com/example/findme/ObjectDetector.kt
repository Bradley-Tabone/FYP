package com.example.findme

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.channels.FileChannel
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Runs yolo26n.tflite on-device via LiteRT (TFLite) to detect
 * glasses, sunglasses, keys, or a wallet.
 *
 * Usage:
 *   1. Call [loadModel] once with the application context.
 *   2. Set a [Listener] via [setListener].
 *   3. Set the active target label via [setTarget] ("wallet", "keys", "glasses", or "sunglasses").
 *   4. Pass each camera frame to [analyze]; the listener is called with every result.
 *
 * Model I/O (YOLO26 TFLite export, end2end=True, 4 classes):
 *   Input  — float32 [1, 640, 640, 3], RGB channel-LAST (NHWC), values in [0, 1]
 *   Output — float32 [1, 300, 6]  (end-to-end, up to 300 detections, no NMS needed)
 *     axis-2 layout: [x1, y1, x2, y2, confidence, class_id]
 *     coordinates are normalised [0,1] or absolute pixels (detected at runtime).
 */
class ObjectDetector {

    private var listener: DetectorContract.Listener? = null
    private var targetLabel: String = ""

    private var interpreter: Interpreter? = null
    private var delegate: Delegate? = null

    // Pre-allocated buffers — reused every frame to avoid per-frame GC pressure.
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())
    private val inputFloatBuffer = inputBuffer.asFloatBuffer()
    private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val floatPixels = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)
    private lateinit var output: Array<Array<FloatArray>>

    // Reused bitmaps + Matrix for the rotation + letterbox stages. Previously allocated
    // a fresh rotated Bitmap, scaled Bitmap, and letterbox Bitmap per frame — ~3-5 ms
    // in allocation + GC pressure. Holding them at class scope eliminates the cost.
    private val rotateMatrix = Matrix()
    private var lastRotation = Int.MIN_VALUE
    private var rotatedBitmap: Bitmap? = null
    private val letterboxBitmap: Bitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas: Canvas = Canvas(letterboxBitmap)
    private val letterboxDstRect = Rect()

    // ── Diagnostic-only telemetry (disable for release by flipping LOG_TIMINGS) ─
    // Logs a per-stage timing breakdown every LOG_EVERY_N frames. Useful for
    // benchmarking the effect of delegate swaps, input resolution changes, or
    // pre-processing optimisations without running Android Studio's profiler.
    private var frameCounter = 0
    private var loggedFrameSize = false

    companion object {
        private const val TAG = "ObjectDetector"
        private const val INPUT_SIZE = 640
        private const val CONFIDENCE_THRESHOLD = 0.3f
        private const val NUM_ANCHORS = 8400   // 80×80 + 40×40 + 20×20 for 640 input
        private const val NUM_CLASSES = 4
        private const val NMS_IOU_THRESHOLD = 0.45f

        private const val LOG_TIMINGS = false
        private const val LOG_EVERY_N = 10

        // A/B toggle — when true, the NNAPI delegate is skipped entirely so the cascade
        // becomes GPU → CPU (XNNPACK). For YOLO26s INT8, NNAPI was partition-loss (silent
        // CPU fallback for unsupported ops + partition overhead = no win). Re-enabled for
        // nano because the smaller graph is more likely to map cleanly onto the NPU.
        // Flip back to true if the logcat shows NNAPI being slower than CPU on this model.
        private const val SKIP_NNAPI = false

        // Maps voice-command labels to TFLite class indices (from data.yaml)
        // Class order: 0=Keys, 1=Sunglasses, 2=Wallet, 3=glasses
        // Note: class 3 (glasses/optical) is a training-only discriminator class — not searchable.
        // Both "sunglasses" and "glasses" voice commands target class 1 (Sunglasses).
        private val LABEL_TO_CLASS_INDEX = mapOf(
            "keys"       to 0,
            "sunglasses" to 1,
            "wallet"     to 2
        )

    }

    fun setListener(listener: DetectorContract.Listener) {
        this.listener = listener
    }

    fun setTarget(label: String) {
        targetLabel = label.lowercase()
    }

    /**
     * Load the TFLite model from assets. Call once before any [analyze] calls.
     *
     * Tries GPU → NNAPI → CPU in order, retrying at the Interpreter level not just
     * the delegate level. This matters for INT8 models whose ops may be accepted by
     * the delegate constructor but rejected when the Interpreter validates the full
     * graph (e.g. INT64 CAST ops that the GPU delegate can't handle).
     *
     * @param context       Application or activity context (used to open assets).
     * @param modelFileName File name inside app/src/main/assets/, e.g. "yolo26s.tflite"
     */
    fun loadModel(context: Context, modelFileName: String) {
        val fd     = context.assets.openFd(modelFileName)
        val mapped = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)

        // Each entry: label to display, factory for the delegate (null = CPU only).
        // We create fresh Interpreter.Options per attempt — you cannot reuse options
        // that have a failed delegate already attached.
        val attempts: List<Pair<String, (() -> Delegate)?>> = listOfNotNull(
            "GPU (FP16, sustained)" to {
                GpuDelegate(GpuDelegate.Options().apply {
                    isPrecisionLossAllowed = true
                    inferencePreference = GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
                })
            },
            if (!SKIP_NNAPI) "NNAPI" to { NnApiDelegate() } else null,
            "CPU (no delegate)" to null
        )

        for ((label, makeDelegate) in attempts) {
            val d: Delegate? = if (makeDelegate != null) {
                try { makeDelegate() }
                catch (t: Throwable) {
                    Log.w(TAG, "$label delegate unavailable: ${t.javaClass.simpleName}")
                    continue
                }
            } else null

            try {
                // availableProcessors() reflects what Android actually allows the app to
                // use. S23 Ultra reports 8, mid-range Snapdragons typically 6, older
                // phones 4. Hardcoding to 4 left half the cores idle on flagships;
                // hardcoding to 8 would over-thread budget devices and add context-switch
                // overhead. Querying at runtime is the portable answer.
                val cpuThreads = Runtime.getRuntime().availableProcessors()
                val options = Interpreter.Options().apply { setNumThreads(cpuThreads) }
                if (d != null) options.addDelegate(d)
                interpreter = Interpreter(mapped, options)
                delegate    = d
                Log.i(TAG, "Model loaded: $modelFileName  via $label")
                val it = interpreter!!
                Log.i(TAG, "  input[0] shape: ${it.getInputTensor(0).shape().toList()}")
                for (i in 0 until it.outputTensorCount) {
                    Log.i(TAG, "  output[$i] shape: ${it.getOutputTensor(i).shape().toList()}")
                }
                val outShape = it.getOutputTensor(0).shape()
                output = Array(1) { Array(outShape[1]) { FloatArray(outShape[2]) } }
                Log.i(TAG, "  output buffer: [1, ${outShape[1]}, ${outShape[2]}]")
                // Prime GPU GLSL/Vulkan compute shaders. Without warmup, the first 5–10
                // camera frames trigger JIT shader compilation and appear as latency spikes
                // (the source of the 160–346 ms spread observed with FAST_SINGLE_ANSWER).
                inputBuffer.rewind()
                repeat(5) { interpreter!!.run(inputBuffer, output) }
                inputBuffer.rewind()
                Log.i(TAG, "  warmup complete (5 runs)")
                return
            } catch (t: Throwable) {
                // Interpreter construction failed with this delegate — release it and
                // try the next option. Log only the first line to keep logcat readable.
                Log.w(TAG, "$label rejected by model: ${t.message?.lineSequence()?.firstOrNull()}")
                d?.close()
            }
        }

        Log.e(TAG, "Failed to load model: $modelFileName — all delegate options exhausted")
    }

    /**
     * Analyse one camera frame. Always closes [image] before returning.
     */
    fun analyze(image: ImageProxy) {
        val tf = interpreter
        if (tf == null) {
            listener?.onResult(null)
            image.close()
            return
        }

        // Idle short-circuit: no active search → don't spend ~500 ms on inference
        // that we're going to throw away. Saves battery and lets the UI thread see
        // null callbacks so the overlay stays cleared.
        if (targetLabel.isEmpty()) {
            listener?.onResult(null)
            image.close()
            return
        }

        val tStart = System.nanoTime()

        try {
            val rawBitmap = image.toBitmap()
            val rotation  = image.imageInfo.rotationDegrees
            val camW = image.width
            val camH = image.height
            image.close()

            if (LOG_TIMINGS && !loggedFrameSize) {
                Log.i(TAG, "camera frame: ${camW}x${camH}  rotation=$rotation")
                loggedFrameSize = true
            }

            // Rotate bitmap to match the display orientation. CameraX delivers
            // frames in the sensor's native orientation (usually landscape),
            // so without this the model's coordinates are rotated relative to
            // the screen — causing swapped width/height and wrong positions.
            val bitmap = rotated(rawBitmap, rotation)

            // Inline inferBitmap with per-stage timestamps so we can see where
            // the frame budget is actually being spent.
            val (scale, padLeft, padTop) = letterboxInto(bitmap)
            val contentW = (bitmap.width  * scale).toInt()
            val contentH = (bitmap.height * scale).toInt()
            val tPrepEnd = System.nanoTime()

            fillInputBuffer(letterboxBitmap)
            val tFillEnd = System.nanoTime()

            tf.run(inputBuffer, output)
            val tInferEnd = System.nanoTime()

            val result = parseOutput(output, padLeft, padTop, contentW, contentH)
            val tEnd = System.nanoTime()

            if (LOG_TIMINGS && (frameCounter++ % LOG_EVERY_N == 0)) {
                val prepMs  = (tPrepEnd  - tStart)    / 1_000_000.0   // YUV→bitmap + rotate + letterbox
                val fillMs  = (tFillEnd  - tPrepEnd)  / 1_000_000.0   // Kotlin per-pixel float fill
                val inferMs = (tInferEnd - tFillEnd)  / 1_000_000.0   // pure tf.run() time
                val postMs  = (tEnd      - tInferEnd) / 1_000_000.0   // parseOutput
                val totalMs = (tEnd      - tStart)    / 1_000_000.0
                val fps     = if (totalMs > 0) 1000.0 / totalMs else 0.0
                Log.d(TAG, "frame: prep=${"%.1f".format(prepMs)}ms  fill=${"%.1f".format(fillMs)}ms  infer=${"%.1f".format(inferMs)}ms  post=${"%.1f".format(postMs)}ms  total=${"%.1f".format(totalMs)}ms  fps=${"%.1f".format(fps)}")
            }

            listener?.onResult(result?.copy(sourceW = bitmap.width, sourceH = bitmap.height))
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            listener?.onResult(null)
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        delegate?.close()
        delegate = null
        rotatedBitmap?.recycle()
        rotatedBitmap = null
        letterboxBitmap.recycle()
    }

    // ── Reused-bitmap rotation + letterbox helpers ────────────────────────────

    /**
     * Returns the input bitmap rotated to display orientation. Reuses a single
     * destination Bitmap and Matrix across frames — the matrix is recomputed only
     * when the rotation angle actually changes (rare; once per device orientation).
     */
    private fun rotated(src: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0) return src

        val swap = (rotation % 180) != 0
        val outW = if (swap) src.height else src.width
        val outH = if (swap) src.width  else src.height

        var dst = rotatedBitmap
        if (dst == null || dst.width != outW || dst.height != outH) {
            dst?.recycle()
            dst = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
            rotatedBitmap = dst
        }

        if (rotation != lastRotation) {
            rotateMatrix.reset()
            rotateMatrix.postRotate(rotation.toFloat(), src.width / 2f, src.height / 2f)
            rotateMatrix.postTranslate((outW - src.width) / 2f, (outH - src.height) / 2f)
            lastRotation = rotation
        }

        Canvas(dst).drawBitmap(src, rotateMatrix, null)
        src.recycle()
        return dst
    }

    /**
     * Letterbox-fits [src] into the pre-allocated [letterboxBitmap]. Returns the
     * scale factor and padding offsets so callers can map model-space coordinates
     * back to source-bitmap coordinates.
     */
    private fun letterboxInto(src: Bitmap): LetterboxParams {
        val scale   = minOf(INPUT_SIZE.toFloat() / src.width, INPUT_SIZE.toFloat() / src.height)
        val newW    = (src.width  * scale).toInt()
        val newH    = (src.height * scale).toInt()
        val padLeft = (INPUT_SIZE - newW) / 2
        val padTop  = (INPUT_SIZE - newH) / 2

        letterboxCanvas.drawColor(Color.rgb(114, 114, 114))
        // Drawing into a destination Rect handles the scale without an intermediate
        // scaled-Bitmap allocation (the old letterbox() called createScaledBitmap).
        letterboxDstRect.set(padLeft, padTop, padLeft + newW, padTop + newH)
        letterboxCanvas.drawBitmap(src, null, letterboxDstRect, null)
        return LetterboxParams(scale, padLeft, padTop)
    }

    private data class LetterboxParams(val scale: Float, val padLeft: Int, val padTop: Int)

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Fill the pre-allocated [inputBuffer] from a 640×640 Bitmap, normalised to [0,1] NHWC.
     *
     * Two deliberate choices for speed:
     *  1. Indexed loop over [pixels] (avoids the iterator overhead that `for (px in pixels)`
     *     incurs on IntArray — which the JIT doesn't always inline cleanly).
     *  2. Bulk-copy via [inputFloatBuffer].put(FloatArray) — a single native memcpy rather than
     *     1.2M individual putFloat() calls through the ByteBuffer proxy.
     * Also uses multiplication by a precomputed 1/255f rather than float division.
     */
    private fun fillInputBuffer(bitmap: Bitmap) {
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val inv255 = 1f / 255f
        var o = 0
        for (i in pixels.indices) {
            val px = pixels[i]
            floatPixels[o++] = ((px shr 16) and 0xFF) * inv255 // R
            floatPixels[o++] = ((px shr 8)  and 0xFF) * inv255 // G
            floatPixels[o++] = ( px         and 0xFF) * inv255 // B
        }
        inputFloatBuffer.rewind()
        inputFloatBuffer.put(floatPixels)
        inputBuffer.rewind()
    }

    /**
     * Parse the raw TFLite output (nms=False) into a [DetectorContract.DetectionResult].
     *
     * Expected layout: [1, 8400, 8]
     *   axis-2 indices: 0=cx, 1=cy, 2=w, 3=h (normalised [0,1] in 640×640 space),
     *                   4..7 = sigmoid class probabilities (Keys, Sunglasses, Wallet, glasses)
     *
     * NMS is applied in Kotlin: candidates for the target class are collected, sorted by
     * confidence, then deduplicated with greedy IoU suppression (threshold 0.45).
     */
    private fun parseOutput(
        output: Array<Array<FloatArray>>,
        padLeft: Int,
        padTop: Int,
        contentW: Int,
        contentH: Int
    ): DetectorContract.DetectionResult? {
        val classIndex = LABEL_TO_CLASS_INDEX[targetLabel] ?: run {
            Log.w(TAG, "Unknown target label: '$targetLabel'")
            return null
        }

        val anchors = output[0]  // [8400][8]
        val targetCands = ArrayList<FloatArray>(64)

        for (a in anchors) {
            val x1: Float; val y1: Float; val x2: Float; val y2: Float
            val tConf: Float

            if (a.size == 6) {
                // End-to-end NMS-free format: [x1, y1, x2, y2, confidence, class_id] (normalised)
                x1 = a[0] * INPUT_SIZE;  y1 = a[1] * INPUT_SIZE
                x2 = a[2] * INPUT_SIZE;  y2 = a[3] * INPUT_SIZE
                tConf = if (a[5].toInt() == classIndex) a[4] else 0f
            } else {
                // Raw anchor format: [cx, cy, w, h, class0_score, class1_score, ...]
                val cxA = a[0] * INPUT_SIZE;  val cyA = a[1] * INPUT_SIZE
                val wA  = a[2] * INPUT_SIZE;  val hA  = a[3] * INPUT_SIZE
                x1 = cxA - wA * 0.5f;        y1 = cyA - hA * 0.5f
                x2 = cxA + wA * 0.5f;        y2 = cyA + hA * 0.5f
                tConf = a[4 + classIndex]
            }

            if (tConf > CONFIDENCE_THRESHOLD) targetCands.add(floatArrayOf(x1, y1, x2, y2, tConf))
        }

        val bestTarget = greedyNms(targetCands) ?: return null

        val cxAbs = (bestTarget[0] + bestTarget[2]) * 0.5f
        val cyAbs = (bestTarget[1] + bestTarget[3]) * 0.5f
        val wAbs  = bestTarget[2] - bestTarget[0]
        val hAbs  = bestTarget[3] - bestTarget[1]
        return DetectorContract.DetectionResult(
            label          = targetLabel,
            confidence     = bestTarget[4],
            normalizedX    = ((cxAbs - padLeft) / contentW).coerceIn(0f, 1f),
            normalizedY    = ((cyAbs - padTop)  / contentH).coerceIn(0f, 1f),
            normalizedArea = (wAbs * hAbs / (contentW * contentH).toFloat()).coerceIn(0f, 1f),
            normalizedW    = (wAbs / contentW).coerceIn(0f, 1f),
            normalizedH    = (hAbs / contentH).coerceIn(0f, 1f)
        )
    }

    /**
     * Greedy NMS on a list of [x1, y1, x2, y2, conf] boxes.
     * Returns the highest-confidence box after suppression, or null if the list is empty.
     */
    private fun greedyNms(candidates: ArrayList<FloatArray>): FloatArray? {
        if (candidates.isEmpty()) return null
        candidates.sortByDescending { it[4] }
        val suppressed = BooleanArray(candidates.size)
        var best: FloatArray? = null

        for (i in candidates.indices) {
            if (suppressed[i]) continue
            val a = candidates[i]
            if (best == null) best = a
            for (j in i + 1 until candidates.size) {
                if (suppressed[j]) continue
                val b = candidates[j]
                val ix1 = maxOf(a[0], b[0]);  val iy1 = maxOf(a[1], b[1])
                val ix2 = minOf(a[2], b[2]);  val iy2 = minOf(a[3], b[3])
                val iw  = maxOf(0f, ix2 - ix1)
                val ih  = maxOf(0f, iy2 - iy1)
                val inter = iw * ih
                val union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
                if (union > 0f && inter / union > NMS_IOU_THRESHOLD) suppressed[j] = true
            }
        }
        return best
    }
}
