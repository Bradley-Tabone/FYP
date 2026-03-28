package com.example.findme

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import kotlin.math.sqrt

/**
 * Transparent overlay drawn on top of the camera preview.
 * Renders a bounding box and confidence label for the current detection result.
 * Non-interactive — touch events pass through to the button beneath.
 *
 * Call [updateDetection] from the main thread with each new [ObjectDetector.DetectionResult]
 * (or null to clear). The box coordinates are approximated from the normalised centre-point
 * and area, assuming a roughly square object.
 */
class BoundingBoxOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detection: ObjectDetector.DetectionResult? = null

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val labelPaint = Paint().apply {
        color = Color.GREEN
        textSize = 48f
        isAntiAlias = true
    }

    private val labelBgPaint = Paint().apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
    }

    /** Update the displayed detection. Pass null to clear the overlay. Must be called on the main thread. */
    fun updateDetection(result: ObjectDetector.DetectionResult?) {
        detection = result
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val result = detection ?: return

        val w = width.toFloat()
        val h = height.toFloat()

        // Approximate box from centre + area, assuming a roughly square object.
        val side   = sqrt(result.normalizedArea)
        val left   = (result.normalizedX - side / 2f).coerceIn(0f, 1f) * w
        val top    = (result.normalizedY - side / 2f).coerceIn(0f, 1f) * h
        val right  = (result.normalizedX + side / 2f).coerceIn(0f, 1f) * w
        val bottom = (result.normalizedY + side / 2f).coerceIn(0f, 1f) * h

        canvas.drawRect(left, top, right, bottom, boxPaint)

        val label = "${result.label} ${"%.0f".format(result.confidence * 100)}%"
        val textBounds = Rect()
        labelPaint.getTextBounds(label, 0, label.length, textBounds)

        // Place label above the box; fall back to below if the box is near the top edge.
        val labelY = if (top > textBounds.height() + 12f) top - 8f
                     else bottom + textBounds.height() + 8f

        canvas.drawRect(
            left,
            labelY - textBounds.height() - 4f,
            left + textBounds.width() + 16f,
            labelY + 4f,
            labelBgPaint
        )
        canvas.drawText(label, left + 8f, labelY, labelPaint)
    }
}
