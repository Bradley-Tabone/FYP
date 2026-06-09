package com.example.findme

import kotlin.math.abs

 /*
  * Smooths the target box over time.
 * The detector can sometimes flicker or jump between frames, so this class
  * remembers the last target and briefly predicts where it should be during
  * very short missed detections.
  */
class TargetTracker {

    companion object {
        // How much the previous position affects the next smoothed position.
        private const val POS_EMA = 0.6f

        // How much the previous velocity affects the next velocity estimate.
        private const val VEL_EMA = 0.7f

        // Confidence is multiplied by this every time the detector misses.
        private const val CONF_DECAY = 0.85f

        // If confidence drops below this value, the tracker gives up.
        private const val CONF_FLOOR = 0.15f

        // Maximum movement allowed per frame, to avoid wild jumps from one bad detection.
        private const val MAX_VEL = 0.05f
    }

    private var x = 0f; private var y = 0f
    private var vx = 0f; private var vy = 0f
    private var conf = 0f

    // Last known box size. Kept the same during short predictions.
    private var w = 0f; private var h = 0f
    private var area = 0f

    // Extra information copied from the last real detection.
    private var label = ""
    private var sourceW = 0; private var sourceH = 0

    private var hasLock = false

    /*
     * Called when the detector finds the target.
     * Updates position, velocity, confidence, and box information.
     */
    fun onHit(d: DetectorTypes.DetectionResult) {
        if (!hasLock) {
            // First detection: store it directly because there is nothing to smooth yet.
            x = d.normalizedX; y = d.normalizedY
            vx = 0f; vy = 0f
        } else {
            val newVx = (d.normalizedX - x).coerceIn(-MAX_VEL, MAX_VEL)
            val newVy = (d.normalizedY - y).coerceIn(-MAX_VEL, MAX_VEL)
            vx = VEL_EMA * vx + (1f - VEL_EMA) * newVx
            vy = VEL_EMA * vy + (1f - VEL_EMA) * newVy
            x  = POS_EMA * x  + (1f - POS_EMA) * d.normalizedX
            y  = POS_EMA * y  + (1f - POS_EMA) * d.normalizedY
        }

        conf = d.confidence
        w = d.normalizedW; h = d.normalizedH; area = d.normalizedArea
        label = d.label
        sourceW = d.sourceW; sourceH = d.sourceH

        hasLock = true
    }

    /*
     * Called when the detector does not find the target.
     * Predicts the next position briefly, then gives up if confidence gets too low.
     */
    fun onMiss(): DetectorTypes.DetectionResult? {
        if (!hasLock) return null

        x = (x + vx).coerceIn(0f, 1f)
        y = (y + vy).coerceIn(0f, 1f)
        conf *= CONF_DECAY

        if (conf < CONF_FLOOR) {
            reset()
            return null
        }

        // Slow the prediction down slightly because we are guessing without a real detection.
        vx *= 0.95f; vy *= 0.95f
        if (abs(vx) < 1e-4f) vx = 0f
        if (abs(vy) < 1e-4f) vy = 0f

        return current()
    }

    // Return the current smoothed or predicted target as a DetectionResult.
    fun current(): DetectorTypes.DetectionResult? {
        if (!hasLock) return null
        return DetectorTypes.DetectionResult(
            label          = label,
            confidence     = conf,
            normalizedX    = x,
            normalizedY    = y,
            normalizedArea = area,
            normalizedW    = w,
            normalizedH    = h,
            sourceW        = sourceW,
            sourceH        = sourceH
        )
    }

    fun reset() {
        x = 0f; y = 0f; vx = 0f; vy = 0f; conf = 0f
        w = 0f; h = 0f; area = 0f
        label = ""; sourceW = 0; sourceH = 0
        hasLock = false
    }
}
