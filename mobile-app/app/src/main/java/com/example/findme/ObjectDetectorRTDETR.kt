package com.example.findme

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageProxy
import java.nio.FloatBuffer

/**
 * RT-DETR-l detector using ONNX Runtime for Android.
 *
 * RT-DETR cannot be exported to TFLite — its deformable cross-attention uses GridSample
 * ops that onnx2tf cannot convert (confirmed upstream issue, Ultralytics #13111/#15026).
 * ONNX Runtime has native GridSample kernels, so the ONNX model runs directly without
 * any conversion step.
 *
 * Model I/O (Ultralytics RT-DETR ONNX export, imgsz=320, 4 classes):
 *   Input  — float32 [1, 3, 320, 320], RGB channel-first (NCHW), values in [0, 1]
 *   Output — float32 [1, 300, 8]  (post-NMS, up to 300 detections)
 *     axis-2 layout: [cx, cy, w, h, score_Keys, score_Sunglasses, score_Wallet, score_glasses]
 *     Coordinates: normalised [0, 1] in the 320×320 letterboxed input space (cxcywh).
 *     Ultralytics RTDETRPredictor calls xywh2xyxy() externally — the ONNX itself outputs
 *     center-format boxes, so parseOutput() converts cxcywh → xyxy before use.
 */
class ObjectDetectorRTDETR {

    private var listener: DetectorContract.Listener? = null
    @Volatile private var targetLabel: String = ""

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var inputName:  String = "images"
    private var outputName: String = "output0"

    // Pre-allocated NCHW input [3 × 320 × 320] — reused every frame.
    private val pixels     = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val nchwPixels = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
    private val rOffset    = 0
    private val gOffset    = INPUT_SIZE * INPUT_SIZE
    private val bOffset    = 2 * INPUT_SIZE * INPUT_SIZE

    // Reused per-frame: a single rotation Matrix, a single rotated-frame Bitmap, and
    // a single 320×320 letterbox Bitmap. Allocating these per frame was burning real
    // milliseconds before — tobImage() already costs a memcpy, no need to add more.
    private val rotateMatrix = Matrix()
    private var lastRotation = Int.MIN_VALUE
    private var rotatedBitmap: Bitmap? = null
    private val letterboxBitmap: Bitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas: Canvas = Canvas(letterboxBitmap)

    // Diagnostic-only telemetry — flip LOG_TIMINGS to false for release.
    private var frameCounter = 0
    private var loggedFrameSize = false

    companion object {
        private const val TAG = "ObjectDetectorRTDETR"
        private const val INPUT_SIZE = 320
        private const val CONFIDENCE_THRESHOLD = 0.3f
        private const val MAX_DETECTIONS = 300
        private const val NUM_CLASSES = 4
        private const val LOG_TIMINGS = true
        private const val LOG_EVERY_N = 10

        // Diagnostic only — when true, ORT prints every node's EP placement at session
        // create. Used once to confirm whether NNAPI accepts any decoder ops, then
        // turned back off (verbose logging itself is slow and floods logcat).
        private const val LOG_ORT_VERBOSE = false

        private val LABEL_TO_CLASS_INDEX = mapOf(
            "keys"       to 0,
            "sunglasses" to 1,
            "wallet"     to 2
        )
    }

    fun setListener(listener: DetectorContract.Listener) { this.listener = listener }
    fun setTarget(label: String) { targetLabel = label.lowercase() }

    fun loadModel(context: Context, modelFileName: String) {
        try {
            val modelBytes = context.assets.open(modelFileName).readBytes()
            val opts = OrtSession.SessionOptions().apply {
                // Snapdragon 8 Gen 2 has 8 cores (1+4+3); let the CPU EP use all of them
                // when NNAPI rejects ops and we fall back to CPU mid-graph.
                setIntraOpNumThreads(8)

                // Explicit ALL_OPT — constant folding, op fusion, layout transforms.
                // Default already runs ALL_OPT but being explicit prevents surprises.
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

                if (LOG_ORT_VERBOSE) {
                    setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
                    setSessionLogVerbosityLevel(0)
                }

                // EP cascade (ORT walks EPs in registration order per op):
                //   NNAPI   — Android's vendor-neutral neural-network abstraction.
                //             The OEM ships its own NNAPI driver that routes ops to the
                //             on-board NPU/DSP/GPU (Hexagon on Qualcomm, APU on MediaTek,
                //             Exynos NPU on Samsung, etc.) — so this single line gives
                //             us hardware acceleration on every Android phone.
                //   XNNPACK — optimised ARM CPU kernels for ops NNAPI rejects.
                //   CPU (default) — last resort.
                //
                // NNAPI tuning via addConfigEntry (NNAPIFlags enum is absent in
                // onnxruntime-android 1.22 AAR; addConfigEntry is the portable path):
                //   use_fp16     — let NNAPI run an FP32 graph in FP16 internally on devices
                //                  whose NPU only supports FP16. Safe; ORT downcasts weights.
                //   use_nchw     — match our [1,3,H,W] input layout, skip a layout transform.
                //   cpu_disabled — refuse NNAPI's *own* CPU fallback for ops the NPU/DSP
                //                  can't run, so those ops bubble down to XNNPACK instead
                //                  (XNNPACK's ARM kernels usually beat NNAPI-CPU). Without
                //                  this NNAPI silently swallows unsupported ops onto its
                //                  own slower CPU path.
                try {
                    addConfigEntry("ep.nnapi.use_fp16", "1")
                    addConfigEntry("ep.nnapi.use_nchw", "1")
                    addConfigEntry("ep.nnapi.cpu_disabled", "1")
                    addNnapi()
                    Log.i(TAG, "NNAPI execution provider registered")
                } catch (t: Throwable) {
                    Log.w(TAG, "NNAPI EP unavailable (${t.javaClass.simpleName})")
                }
                try {
                    addXnnpack(mapOf("intra_op_num_threads" to "8"))
                    Log.i(TAG, "XNNPACK execution provider registered")
                } catch (t: Throwable) {
                    Log.w(TAG, "XNNPACK EP unavailable (${t.javaClass.simpleName})")
                }
            }
            session    = env.createSession(modelBytes, opts)
            inputName  = session!!.inputNames.iterator().next()
            outputName = session!!.outputNames.iterator().next()
            Log.i(TAG, "Model loaded: $modelFileName")
            Log.i(TAG, "  input:  $inputName  info=${session!!.inputInfo[inputName]}")
            Log.i(TAG, "  output: $outputName  info=${session!!.outputInfo[outputName]}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: $modelFileName", e)
        }
    }

    fun analyze(image: ImageProxy) {
        if (session == null) {
            listener?.onResult(null)
            image.close()
            return
        }
        // Snapshot volatile field once — prevents a race where the main thread calls
        // setTarget("") during the ~600ms inference window, causing parseOutput to see
        // a different (empty) label than the isEmpty check saw at the top of this call.
        val target = targetLabel
        if (target.isEmpty()) {
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

            val bitmap = rotated(rawBitmap, rotation)
            val tPrepEnd = System.nanoTime()

            val (scale, padLeft, padTop) = letterboxInto(bitmap)
            val contentW = (bitmap.width  * scale).toInt()
            val contentH = (bitmap.height * scale).toInt()
            val tLetterEnd = System.nanoTime()

            fillNchwBuffer(letterboxBitmap)
            val tFillEnd = System.nanoTime()

            val result: DetectorContract.DetectionResult?
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(nchwPixels),
                longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            ).use { inputTensor ->
                session!!.run(mapOf(inputName to inputTensor)).use { results ->
                    val tInferEnd = System.nanoTime()
                    val outTensor = results[outputName].get() as OnnxTensor
                    result = parseOutput(target, outTensor.floatBuffer, padLeft, padTop, contentW, contentH)
                    val tEnd = System.nanoTime()

                    if (LOG_TIMINGS && (frameCounter++ % LOG_EVERY_N == 0)) {
                        val prepMs   = (tPrepEnd   - tStart)     / 1_000_000.0  // tobImage + rotate
                        val letterMs = (tLetterEnd - tPrepEnd)   / 1_000_000.0  // letterbox draw
                        val fillMs   = (tFillEnd   - tLetterEnd) / 1_000_000.0  // NCHW pixel fill
                        val inferMs  = (tInferEnd  - tFillEnd)   / 1_000_000.0  // pure ORT.run()
                        val postMs   = (tEnd       - tInferEnd)  / 1_000_000.0  // parseOutput
                        val totalMs  = (tEnd       - tStart)     / 1_000_000.0
                        val fps      = if (totalMs > 0) 1000.0 / totalMs else 0.0
                        Log.d(
                            TAG,
                            "frame: prep=${"%.1f".format(prepMs)}ms  letter=${"%.1f".format(letterMs)}ms  " +
                                "fill=${"%.1f".format(fillMs)}ms  infer=${"%.1f".format(inferMs)}ms  " +
                                "post=${"%.1f".format(postMs)}ms  total=${"%.1f".format(totalMs)}ms  " +
                                "fps=${"%.1f".format(fps)}"
                        )
                    }
                }
            }

            listener?.onResult(result?.copy(sourceW = bitmap.width, sourceH = bitmap.height))
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            listener?.onResult(null)
        }
    }

    fun close() {
        session?.close()
        session = null
        rotatedBitmap?.recycle()
        rotatedBitmap = null
        letterboxBitmap.recycle()
    }

    // ── Core inference ────────────────────────────────────────────────────────

    /**
     * Returns the input bitmap rotated to display orientation. Reuses a single
     * destination Bitmap across frames — the rotation matrix is recomputed only
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
            // Rotate around the source-bitmap centre, then translate so the
            // rotated frame lands at (0,0) of the destination bitmap.
            rotateMatrix.postRotate(rotation.toFloat(), src.width / 2f, src.height / 2f)
            rotateMatrix.postTranslate((outW - src.width) / 2f, (outH - src.height) / 2f)
            lastRotation = rotation
        }

        Canvas(dst).drawBitmap(src, rotateMatrix, null)
        src.recycle()
        return dst
    }

    /** Letterbox-fits [src] into the pre-allocated [letterboxBitmap]. */
    private fun letterboxInto(src: Bitmap): LetterboxParams {
        val scale   = minOf(INPUT_SIZE.toFloat() / src.width, INPUT_SIZE.toFloat() / src.height)
        val newW    = (src.width  * scale).toInt()
        val newH    = (src.height * scale).toInt()
        val padLeft = (INPUT_SIZE - newW) / 2
        val padTop  = (INPUT_SIZE - newH) / 2

        letterboxCanvas.drawColor(Color.rgb(114, 114, 114))
        // Drawing with a destination Rect handles the scaling without an intermediate
        // scaled-Bitmap allocation.
        val dstRect = android.graphics.Rect(padLeft, padTop, padLeft + newW, padTop + newH)
        letterboxCanvas.drawBitmap(src, null, dstRect, null)
        return LetterboxParams(scale, padLeft, padTop)
    }

    private data class LetterboxParams(val scale: Float, val padLeft: Int, val padTop: Int)

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Fill [nchwPixels] as NCHW from a 320×320 Bitmap.
     * ONNX models use channel-first (all R, then all G, then all B) unlike TFLite's NHWC.
     */
    private fun fillNchwBuffer(bitmap: Bitmap) {
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val inv255 = 1f / 255f
        for (i in pixels.indices) {
            val px = pixels[i]
            nchwPixels[rOffset + i] = ((px shr 16) and 0xFF) * inv255
            nchwPixels[gOffset + i] = ((px shr 8)  and 0xFF) * inv255
            nchwPixels[bOffset + i] = ( px         and 0xFF) * inv255
        }
    }

    /**
     * Output layout: [1, 300, 8] flat buffer.
     * Each detection: [cx, cy, w, h, score_Keys, score_Sunglasses, score_Wallet, score_glasses]
     * Coordinates are normalised [0,1] in the 320×320 letterboxed space (cxcywh format).
     * Ultralytics RTDETRPredictor.postprocess() calls xywh2xyxy() on these before use —
     * the ONNX model itself outputs raw center-format coordinates, not corner-format.
     */
    private fun parseOutput(
        target: String,
        buffer: FloatBuffer,
        padLeft: Int, padTop: Int,
        contentW: Int, contentH: Int
    ): DetectorContract.DetectionResult? {
        val classIndex = LABEL_TO_CLASS_INDEX[target] ?: run {
            Log.w(TAG, "Unknown target label: '$target'")
            return null
        }

        var best: DetectorContract.DetectionResult? = null
        var bestConf = CONFIDENCE_THRESHOLD

        // Debug: track the highest target-class score seen, even if below threshold or beaten by another class
        var debugMaxTargetConf = 0f
        var debugMaxAnyConf    = 0f
        var debugTopClassForMax = -1

        for (i in 0 until MAX_DETECTIONS) {
            val base = i * (4 + NUM_CLASSES)
            val conf = buffer.get(base + 4 + classIndex)
            if (conf > debugMaxTargetConf) debugMaxTargetConf = conf
            val maxScore = (0 until NUM_CLASSES).maxOf { buffer.get(base + 4 + it) }
            if (maxScore > debugMaxAnyConf) {
                debugMaxAnyConf = maxScore
                debugTopClassForMax = (0 until NUM_CLASSES).maxByOrNull { buffer.get(base + 4 + it) } ?: -1
            }
            if (conf <= bestConf) continue
            // Reject if any other class scored higher
            if (conf < maxScore) continue

            bestConf = conf
            // Convert cxcywh (model output) → xyxy in letterboxed pixel space
            val cx = buffer.get(base)     * INPUT_SIZE
            val cy = buffer.get(base + 1) * INPUT_SIZE
            val bw = buffer.get(base + 2) * INPUT_SIZE
            val bh = buffer.get(base + 3) * INPUT_SIZE
            val x1 = cx - bw / 2f
            val y1 = cy - bh / 2f
            val x2 = cx + bw / 2f
            val y2 = cy + bh / 2f
            best = DetectorContract.DetectionResult(
                label          = target,
                confidence     = conf,
                normalizedX    = (((x1 + x2) / 2f - padLeft) / contentW).coerceIn(0f, 1f),
                normalizedY    = (((y1 + y2) / 2f - padTop)  / contentH).coerceIn(0f, 1f),
                normalizedArea = ((x2 - x1) * (y2 - y1) / (contentW * contentH).toFloat()).coerceIn(0f, 1f),
                normalizedW    = ((x2 - x1) / contentW).coerceIn(0f, 1f),
                normalizedH    = ((y2 - y1) / contentH).coerceIn(0f, 1f)
            )
        }
        Log.d(TAG, "parse: target='$target'(cls=$classIndex) maxTargetConf=${"%.3f".format(debugMaxTargetConf)} maxAnyConf=${"%.3f".format(debugMaxAnyConf)} topClass=$debugTopClassForMax found=${best != null}")
        return best
    }
}
