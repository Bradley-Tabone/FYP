package com.example.findme

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import kotlin.math.sqrt

/*
 * Transparent view placed over the camera preview.
 * It does not detect objects itself. It only draws the latest DetectionResult
 * as a green box with a confidence label.
 */
class BoundingBoxOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // Current detection to draw. Null means no box is shown.
    private var detection: DetectorTypes.DetectionResult? = null

    // Styling for the green box outline.
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    // Styling for the green label text.
    private val labelPaint = Paint().apply {
        color = Color.GREEN
        textSize = 48f
        isAntiAlias = true
    }

    // Styling for the dark background behind the label.
    private val labelBgPaint = Paint().apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
    }

    // Reused rectangle used to measure how much space the label text needs.
    private val textBounds = Rect()

    // Store the new detection and ask Android to redraw this view.
    fun updateDetection(result: DetectorTypes.DetectionResult?) {
        detection = result
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val result = detection ?: return

        val viewW = width.toFloat()
        val viewH = height.toFloat()

        // Use the real box width and height if the detection has them.
        val bw = if (result.normalizedW > 0f) result.normalizedW else sqrt(result.normalizedArea)
        val bh = if (result.normalizedH > 0f) result.normalizedH else sqrt(result.normalizedArea)

        /*
         * The detection position is stored as normalized camera coordinates from 0 to 1.
         * Android drawing needs real screen pixels, so the code below converts the
         * detection box into left, top, right, and bottom pixel positions.
         */
        val left:   Float
        val top:    Float
        val right:  Float
        val bottom: Float

        if (result.sourceW > 0 && result.sourceH > 0) {
            val frameAspect = result.sourceW.toFloat() / result.sourceH
            val viewAspect  = viewW / viewH
            val scale: Float
            val ox: Float
            val oy: Float

            if (frameAspect > viewAspect) {
                // Wider frame: fill the view height and account for left/right cropping.
                scale = viewH / result.sourceH
                val scaledW = result.sourceW * scale
                ox = (scaledW - viewW) / 2f
                oy = 0f
            } else {
                // Taller frame: fill the view width and account for top/bottom cropping.
                scale = viewW / result.sourceW
                val scaledH = result.sourceH * scale
                ox = 0f
                oy = (scaledH - viewH) / 2f
            }

            val scaledFrameW = result.sourceW * scale
            val scaledFrameH = result.sourceH * scale

            // Convert normalized frame coordinates into visible screen pixels.
            left   = ((result.normalizedX - bw / 2f) * scaledFrameW - ox).coerceIn(0f, viewW)
            top    = ((result.normalizedY - bh / 2f) * scaledFrameH - oy).coerceIn(0f, viewH)
            right  = ((result.normalizedX + bw / 2f) * scaledFrameW - ox).coerceIn(0f, viewW)
            bottom = ((result.normalizedY + bh / 2f) * scaledFrameH - oy).coerceIn(0f, viewH)
        } else {
            // Fallback if the detection does not include the original camera frame size.
            left   = (result.normalizedX - bw / 2f).coerceIn(0f, 1f) * viewW
            top    = (result.normalizedY - bh / 2f).coerceIn(0f, 1f) * viewH
            right  = (result.normalizedX + bw / 2f).coerceIn(0f, 1f) * viewW
            bottom = (result.normalizedY + bh / 2f).coerceIn(0f, 1f) * viewH
        }

        canvas.drawRect(left, top, right, bottom, boxPaint)

        val label = "${result.label} ${(result.confidence * 100).toInt()}%"
        labelPaint.getTextBounds(label, 0, label.length, textBounds)

        // Place the label above the box, unless the box is too close to the top.
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
