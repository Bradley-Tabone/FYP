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
 * Call [updateDetection] from the main thread with each new [DetectorContract.DetectionResult]
 * (or null to clear). When normalizedW/H are provided the actual box shape is drawn;
 * otherwise falls back to a square approximation from centre-point + area.
 *
 * The overlay accounts for PreviewView's FILL_CENTER scaling by using the source
 * frame dimensions (sourceW/sourceH) from the detection result to compute the
 * crop offset. This ensures the bounding box aligns with the visible preview.
 */
class BoundingBoxOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detection: DetectorContract.DetectionResult? = null

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
    fun updateDetection(result: DetectorContract.DetectionResult?) {
        detection = result
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val result = detection ?: return

        val viewW = width.toFloat()
        val viewH = height.toFloat()

        // Use actual box width/height when available; fall back to square approximation.
        val bw = if (result.normalizedW > 0f) result.normalizedW else sqrt(result.normalizedArea)
        val bh = if (result.normalizedH > 0f) result.normalizedH else sqrt(result.normalizedArea)

        // Compute FILL_CENTER crop offset.
        // PreviewView scales the camera frame to fill the view and crops the excess.
        // Model coordinates are in [0,1] of the full frame, but the view only shows
        // the cropped centre. We map frame-normalised coords to view pixels correctly.
        val left:   Float
        val top:    Float
        val right:  Float
        val bottom: Float

        if (result.sourceW > 0 && result.sourceH > 0) {
            val frameAspect = result.sourceW.toFloat() / result.sourceH
            val viewAspect  = viewW / viewH
            val scale: Float
            val ox: Float   // horizontal crop offset in view pixels
            val oy: Float   // vertical   crop offset in view pixels

            if (frameAspect > viewAspect) {
                // Frame is wider than view → scaled to fill height, crop left/right
                scale = viewH / result.sourceH
                val scaledW = result.sourceW * scale
                ox = (scaledW - viewW) / 2f
                oy = 0f
            } else {
                // Frame is taller than view → scaled to fill width, crop top/bottom
                scale = viewW / result.sourceW
                val scaledH = result.sourceH * scale
                ox = 0f
                oy = (scaledH - viewH) / 2f
            }

            val scaledFrameW = result.sourceW * scale
            val scaledFrameH = result.sourceH * scale

            // Map normalised frame coords → view pixels
            left   = ((result.normalizedX - bw / 2f) * scaledFrameW - ox).coerceIn(0f, viewW)
            top    = ((result.normalizedY - bh / 2f) * scaledFrameH - oy).coerceIn(0f, viewH)
            right  = ((result.normalizedX + bw / 2f) * scaledFrameW - ox).coerceIn(0f, viewW)
            bottom = ((result.normalizedY + bh / 2f) * scaledFrameH - oy).coerceIn(0f, viewH)
        } else {
            // Fallback: no source dimensions → assume 1:1 mapping (legacy behaviour)
            left   = (result.normalizedX - bw / 2f).coerceIn(0f, 1f) * viewW
            top    = (result.normalizedY - bh / 2f).coerceIn(0f, 1f) * viewH
            right  = (result.normalizedX + bw / 2f).coerceIn(0f, 1f) * viewW
            bottom = (result.normalizedY + bh / 2f).coerceIn(0f, 1f) * viewH
        }

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
