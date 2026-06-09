package com.example.findme

import android.content.Context
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

/*
 * Runs the TFLite object detection model on camera frames.
 * It loads the model, prepares each camera image into 640x640 RGB input,
 * runs inference, then reports the parsed detection result through the listener.
 */
class ObjectDetector {

    private var listener: DetectorTypes.Listener? = null
    private var targetLabel: String = ""

    private var interpreter: Interpreter? = null
    private var delegate: Delegate? = null

    // Reused input buffers for the model's 640x640 RGB float image.
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())
    private val inputFloatBuffer = inputBuffer.asFloatBuffer()
    private val floatPixels = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)
    private lateinit var output: Array<Array<FloatArray>>

    /*
     * Reused arrays for the camera image's Y, U, and V planes.
     * They are sized on the first frame and reused so each frame does not
     * need fresh byte arrays.
     */
    private var yArray: ByteArray? = null
    private var uArray: ByteArray? = null
    private var vArray: ByteArray? = null

    // Used only for timing/debug logs.
    private var frameCounter = 0
    private var loggedFrameSize = false

    companion object {
        private const val TAG = "ObjectDetector"
        private const val INPUT_SIZE = 640

        private const val LOG_TIMINGS = true
        private const val LOG_EVERY_N = 10

        // Set true to skip NNAPI and only try GPU, then CPU.
        private const val SKIP_NNAPI = false
    }

    fun setListener(listener: DetectorTypes.Listener) {
        this.listener = listener
    }

    fun setTarget(label: String) {
        targetLabel = label.lowercase()
    }

    /*
     * Loads the TFLite model from assets.
     * It tries GPU, then NNAPI, then CPU, and keeps the first option that
     * successfully builds and warms up.
     */
    fun loadModel(context: Context, modelFileName: String) {
        val fd     = context.assets.openFd(modelFileName)
        val mapped = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)

        // Use the number of processor cores Android makes available to the app.
        val cpuThreads = Runtime.getRuntime().availableProcessors()

        /*
         * Each attempt describes one way to run the model.
         * A null delegate means plain CPU execution.
         */
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

            // Temporary interpreter used while checking whether this attempt works.
            var candidate: Interpreter? = null
            try {
                val options = Interpreter.Options().apply { setNumThreads(cpuThreads) }
                if (d != null) options.addDelegate(d)
                candidate = Interpreter(mapped, options)
                Log.i(TAG, "Model loaded: $modelFileName  via $label")
                Log.i(TAG, "  input[0] shape: ${candidate.getInputTensor(0).shape().toList()}")
                for (i in 0 until candidate.outputTensorCount) {
                    Log.i(TAG, "  output[$i] shape: ${candidate.getOutputTensor(i).shape().toList()}")
                }
                val outShape = candidate.getOutputTensor(0).shape()
                output = Array(1) { Array(outShape[1]) { FloatArray(outShape[2]) } }
                Log.i(TAG, "  output buffer: [1, ${outShape[1]}, ${outShape[2]}]")
                // Run a few fake inferences so the first real camera frames are less delayed.
                inputBuffer.rewind()
                repeat(5) { candidate.run(inputBuffer, output) }
                inputBuffer.rewind()
                Log.i(TAG, "  warmup complete (5 runs)")
                // Store this interpreter only after it has successfully warmed up.
                interpreter = candidate
                delegate    = d
                return
            } catch (t: Throwable) {
                // If this attempt fails, clean it up and try the next option.
                Log.w(TAG, "$label rejected by model: ${t.message?.lineSequence()?.firstOrNull()}")
                candidate?.close()
                d?.close()
            }
        }

        Log.e(TAG, "Failed to load model: $modelFileName — all delegate options exhausted")
    }

    // Analyse one camera frame and always close it before returning.
    fun analyze(image: ImageProxy) {
        val tf = interpreter
        if (tf == null) {
            listener?.onResult(null)
            image.close()
            return
        }

        /*
         * If no object is being searched for, skip inference.
         * This saves battery and keeps the overlay cleared.
         */
        if (targetLabel.isEmpty()) {
            listener?.onResult(null)
            image.close()
            return
        }

        val tStart = System.nanoTime()

        try {
            val rotation = image.imageInfo.rotationDegrees

            if (LOG_TIMINGS && !loggedFrameSize) {
                Log.i(TAG, "camera frame: ${image.width}x${image.height}  rotation=$rotation")
                loggedFrameSize = true
            }

            /*
             * Prepare the camera frame for the model:
             * rotate it, fit it into 640x640, convert YUV to RGB, normalize it,
             * and calculate brightness for the torch logic.
             */
            val params = yuvToFloatDirect(image, rotation)
            val tDecodeEnd = System.nanoTime()

            val rotW     = if (rotation % 180 != 0) image.height else image.width
            val rotH     = if (rotation % 180 != 0) image.width  else image.height
            val contentW = (rotW * params.scale).toInt()
            val contentH = (rotH * params.scale).toInt()

            tf.run(inputBuffer, output)
            val tInferEnd = System.nanoTime()

            val result = parseOutput(output, params.padLeft, params.padTop, contentW, contentH)
            val tEnd = System.nanoTime()

            if (LOG_TIMINGS && (frameCounter++ % LOG_EVERY_N == 0)) {
                val decodeMs = (tDecodeEnd - tStart)     / 1_000_000.0
                val inferMs  = (tInferEnd  - tDecodeEnd) / 1_000_000.0
                val postMs   = (tEnd       - tInferEnd)  / 1_000_000.0
                val totalMs  = (tEnd       - tStart)     / 1_000_000.0
                val fps      = if (totalMs > 0) 1000.0 / totalMs else 0.0
                Log.d(TAG, "frame: decode=${"%.1f".format(decodeMs)}ms  infer=${"%.1f".format(inferMs)}ms  post=${"%.1f".format(postMs)}ms  total=${"%.1f".format(totalMs)}ms  fps=${"%.1f".format(fps)}")
            }

            // Send brightness first, then the parsed detection result.
            listener?.onLuma(params.meanLuma)
            listener?.onResult(result?.copy(sourceW = rotW, sourceH = rotH))
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            listener?.onResult(null)
        } finally {
            image.close()
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        delegate?.close()
        delegate = null
    }

    /*
     * Converts a CameraX YUV frame into the RGB float buffer required by the model.
     * This function also applies the rotation correction, letterbox padding,
     * and mean brightness calculation.
     */
    private fun yuvToFloatDirect(image: ImageProxy, rotation: Int): LetterboxParams {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        // Rewind before reading so each plane is copied from the start.
        val yBuf = yPlane.buffer.also { it.rewind() }
        val uBuf = uPlane.buffer.also { it.rewind() }
        val vBuf = vPlane.buffer.also { it.rewind() }

        // Resize the reused YUV arrays only if the camera frame size changes.
        val ySize = yBuf.remaining(); if (yArray?.size != ySize) yArray = ByteArray(ySize)
        val uSize = uBuf.remaining(); if (uArray?.size != uSize) uArray = ByteArray(uSize)
        val vSize = vBuf.remaining(); if (vArray?.size != vSize) vArray = ByteArray(vSize)
        yBuf.get(yArray!!); uBuf.get(uArray!!); vBuf.get(vArray!!)

        val y = yArray!!; val u = uArray!!; val v = vArray!!
        val yRowStride  = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixStride = uPlane.pixelStride

        val srcW = image.width
        val srcH = image.height
        val rotW = if (rotation % 180 != 0) srcH else srcW
        val rotH = if (rotation % 180 != 0) srcW else srcH

        val scale    = minOf(INPUT_SIZE.toFloat() / rotW, INPUT_SIZE.toFloat() / rotH)
        val contentW = (rotW * scale).toInt()
        val contentH = (rotH * scale).toInt()
        val padLeft  = (INPUT_SIZE - contentW) / 2
        val padTop   = (INPUT_SIZE - contentH) / 2

        val inv255   = 1f / 255f
        val grayNorm = 114f * inv255   // Normalized grey value used for letterbox padding.
        val scaleInv = 1f / scale

        // Fill the top letterbox padding.
        fillGreyRows(0, padTop, grayNorm)

        /*
         * Decode the real image area.
         * Each branch tells decodeBand how to map corrected pixels back to
         * source pixels for the current rotation.
         */
        val lumaSum: Long = when (rotation) {
            90 -> decodeBand(
                contentW, contentH, padLeft, padTop, scaleInv, inv255, grayNorm,
                y, u, v, yRowStride, uvRowStride, uvPixStride,
                sxFn = { _, ry -> ry },
                syFn = { rx, _ -> srcH - 1 - rx }
            )
            180 -> decodeBand(
                contentW, contentH, padLeft, padTop, scaleInv, inv255, grayNorm,
                y, u, v, yRowStride, uvRowStride, uvPixStride,
                sxFn = { rx, _ -> srcW - 1 - rx },
                syFn = { _, ry -> srcH - 1 - ry }
            )
            270 -> decodeBand(
                contentW, contentH, padLeft, padTop, scaleInv, inv255, grayNorm,
                y, u, v, yRowStride, uvRowStride, uvPixStride,
                sxFn = { _, ry -> srcW - 1 - ry },
                syFn = { rx, _ -> rx }
            )
            else -> decodeBand(
                contentW, contentH, padLeft, padTop, scaleInv, inv255, grayNorm,
                y, u, v, yRowStride, uvRowStride, uvPixStride,
                sxFn = { rx, _ -> rx },
                syFn = { _, ry -> ry }
            )
        }

        // Fill the bottom letterbox padding.
        fillGreyRows(padTop + contentH, INPUT_SIZE - padTop - contentH, grayNorm)

        // Copy the prepared pixels into the direct model input buffer.
        inputFloatBuffer.rewind()
        inputFloatBuffer.put(floatPixels)
        inputBuffer.rewind()

        val sampled  = contentW * contentH
        val meanLuma = if (sampled > 0) (lumaSum / sampled).toInt() else 0
        return LetterboxParams(scale, padLeft, padTop, meanLuma)
    }

    // Fill a group of model-input rows with the grey letterbox padding value.
    private fun fillGreyRows(startRow: Int, rowCount: Int, grayNorm: Float) {
        if (rowCount <= 0) return
        val start = startRow * INPUT_SIZE * 3
        val end   = start + rowCount * INPUT_SIZE * 3
        var fi = start
        while (fi < end) {
            floatPixels[fi++] = grayNorm
        }
    }

    /*
     * Decodes the real image area into RGB floats.
     * sxFn and syFn decide which source pixel to read after rotation correction.
     */
    private inline fun decodeBand(
        contentW: Int, contentH: Int, padLeft: Int, padTop: Int,
        scaleInv: Float, inv255: Float, grayNorm: Float,
        y: ByteArray, u: ByteArray, v: ByteArray,
        yRowStride: Int, uvRowStride: Int, uvPixStride: Int,
        crossinline sxFn: (rx: Int, ry: Int) -> Int,
        crossinline syFn: (rx: Int, ry: Int) -> Int
    ): Long {
        var lumaSum = 0L
        val rightPadStart = padLeft + contentW
        val rightPadCount = INPUT_SIZE - rightPadStart

        for (oy in 0 until contentH) {
            val ry = (oy * scaleInv).toInt()
            var fi = (padTop + oy) * INPUT_SIZE * 3

            // Fill left letterbox padding for this row.
            var lp = padLeft
            while (lp > 0) {
                floatPixels[fi++] = grayNorm
                floatPixels[fi++] = grayNorm
                floatPixels[fi++] = grayNorm
                lp--
            }

            // Decode the real image pixels for this row.
            for (ox in 0 until contentW) {
                val rx = (ox * scaleInv).toInt()
                val sx = sxFn(rx, ry)
                val sy = syFn(rx, ry)

                val yVal  = y[sy * yRowStride + sx].toInt() and 0xFF
                val uvOff = (sy / 2) * uvRowStride + (sx / 2) * uvPixStride
                val uVal  = (u[uvOff].toInt() and 0xFF) - 128
                val vVal  = (v[uvOff].toInt() and 0xFF) - 128

                // Convert YUV to normalized RGB.
                floatPixels[fi++] = (yVal + 1.402f    * vVal).coerceIn(0f, 255f) * inv255
                floatPixels[fi++] = (yVal - 0.344136f * uVal - 0.714136f * vVal).coerceIn(0f, 255f) * inv255
                floatPixels[fi++] = (yVal + 1.772f    * uVal).coerceIn(0f, 255f) * inv255

                lumaSum += yVal
            }

            // Fill right letterbox padding for this row.
            var rp = rightPadCount
            while (rp > 0) {
                floatPixels[fi++] = grayNorm
                floatPixels[fi++] = grayNorm
                floatPixels[fi++] = grayNorm
                rp--
            }
        }
        return lumaSum
    }

    private data class LetterboxParams(
        val scale: Float,
        val padLeft: Int,
        val padTop: Int,
        val meanLuma: Int
    )

    // Convert the raw model output into the shared DetectionResult type.
    private fun parseOutput(
        output: Array<Array<FloatArray>>,
        padLeft: Int,
        padTop: Int,
        contentW: Int,
        contentH: Int
    ): DetectorTypes.DetectionResult? {
        return DetectionOutputParser.parse(targetLabel, output, padLeft, padTop, contentW, contentH)
    }
}
