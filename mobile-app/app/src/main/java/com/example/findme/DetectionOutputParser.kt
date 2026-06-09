package com.example.findme

/*
 * Converts the model's raw detection numbers into the app's DetectionResult format.
 * This parser only supports the end-to-end output shape:
 * [x1, y1, x2, y2, confidence, class_id].
 */
object DetectionOutputParser {

    private const val INPUT_SIZE = 640f
    private const val CONFIDENCE_THRESHOLD = 0.3f

    private val labelToClassIndex = mapOf(
        "keys" to 0,
        "sunglasses" to 1,
        "wallet" to 2
    )

    fun parse(
        targetLabel: String,
        output: Array<Array<FloatArray>>,
        padLeft: Int,
        padTop: Int,
        contentW: Int,
        contentH: Int
    ): DetectorTypes.DetectionResult? {
        val classIndex = labelToClassIndex[targetLabel.lowercase()] ?: return null
        val tensor = output.firstOrNull() ?: return null
        if (tensor.isEmpty() || contentW <= 0 || contentH <= 0) return null

        val accessor = OutputAccessor.from(tensor) ?: return null

        val best = if (accessor.channelCount == 6) {
            pickEndToEndBest(accessor, classIndex)
        } else {
            null
        } ?: return null

        val cxAbs = (best[0] + best[2]) * 0.5f
        val cyAbs = (best[1] + best[3]) * 0.5f
        val wAbs = best[2] - best[0]
        val hAbs = best[3] - best[1]

        return DetectorTypes.DetectionResult(
            label = targetLabel.lowercase(),
            confidence = best[4],
            normalizedX = ((cxAbs - padLeft) / contentW).coerceIn(0f, 1f),
            normalizedY = ((cyAbs - padTop) / contentH).coerceIn(0f, 1f),
            normalizedArea = (wAbs * hAbs / (contentW * contentH).toFloat()).coerceIn(0f, 1f),
            normalizedW = (wAbs / contentW).coerceIn(0f, 1f),
            normalizedH = (hAbs / contentH).coerceIn(0f, 1f)
        )
    }

    /*
     * Loops through the model detections and keeps the highest-confidence one
     * that matches the object the user is searching for.
     */
    private fun pickEndToEndBest(
        accessor: OutputAccessor,
        classIndex: Int
    ): FloatArray? {
        var bestConf = CONFIDENCE_THRESHOLD
        var bestIdx = -1
        for (i in 0 until accessor.anchorCount) {
            // Channel 5 stores the detected class id.
            if (accessor.valueAt(i, 5).toInt() != classIndex) continue
            val conf = accessor.valueAt(i, 4)
            if (conf > bestConf) {
                bestConf = conf
                bestIdx = i
            }
        }
        if (bestIdx < 0) return null

        val x1 = accessor.valueAt(bestIdx, 0)
        val y1 = accessor.valueAt(bestIdx, 1)
        val x2 = accessor.valueAt(bestIdx, 2)
        val y2 = accessor.valueAt(bestIdx, 3)
        val scale = coordinateScale(x1, y1, x2, y2)
        scratch[0] = x1 * scale
        scratch[1] = y1 * scale
        scratch[2] = x2 * scale
        scratch[3] = y2 * scale
        scratch[4] = bestConf
        return scratch
    }

    private fun coordinateScale(c0: Float, c1: Float, c2: Float, c3: Float): Float {
        // Small coordinate values are treated as normalized 0..1 values.
        return if (maxOf(c0, c1, c2, c3) <= 1.5f) INPUT_SIZE else 1f
    }

    // Reused temporary array holding [x1, y1, x2, y2, confidence].
    private val scratch = FloatArray(5)

    /*
     * Lets the parser read model output the same way whether the tensor is
     * [detections][values] or [values][detections].
     */
    private sealed interface OutputAccessor {
        val anchorCount: Int
        val channelCount: Int
        fun valueAt(anchorIndex: Int, channelIndex: Int): Float

        class RowMajor(private val rows: Array<FloatArray>) : OutputAccessor {
            override val anchorCount: Int = rows.size
            override val channelCount: Int = rows[0].size
            override fun valueAt(anchorIndex: Int, channelIndex: Int): Float =
                rows[anchorIndex][channelIndex]
        }

        class Transposed(private val channels: Array<FloatArray>) : OutputAccessor {
            override val anchorCount: Int = channels[0].size
            override val channelCount: Int = channels.size
            override fun valueAt(anchorIndex: Int, channelIndex: Int): Float =
                channels[channelIndex][anchorIndex]
        }

        companion object {
            fun from(tensor: Array<FloatArray>): OutputAccessor? {
                val firstWidth = tensor.firstOrNull()?.size ?: return null
                return when {
                    firstWidth == 6 -> RowMajor(tensor)
                    tensor.size == 6 -> Transposed(tensor)
                    else -> null
                }
            }
        }
    }
}
