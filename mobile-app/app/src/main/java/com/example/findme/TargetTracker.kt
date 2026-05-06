package com.example.findme

import kotlin.math.abs

/**
 * Single-target temporal smoother with constant-velocity occlusion bridging.
 *
 * Sits between the raw detector callback and MainActivity's UI/audio update.
 * Replaces the simple `MISS_HOLD_FRAMES` freeze with:
 *   - position EMA on hits  → smooths jitter, no popping when detection returns
 *   - velocity estimate     → predicted box keeps moving with the camera/object during a miss
 *   - confidence decay      → predictions self-expire after roughly 12 frames (~330 ms @ 30 fps)
 *
 * Why not a Kalman filter / SORT / ByteTrack: those are designed for multi-target
 * tracking with re-identification. We only ever track one target (the user's spoken
 * choice), so the matrix machinery is pure overhead. ~5 floats of state, sub-ms cost.
 *
 * Limitations (worth raising in the meeting):
 *   - Bridges short occlusions only (under ~½ second). Anything longer falls below
 *     CONF_FLOOR and the tracker resets — by design.
 *   - Assumes roughly linear short-term motion. Sudden direction changes during the
 *     occlusion gap mis-predict, but the confidence decay catches that quickly.
 *   - This is a UX smoother, not a model improvement. mAP numbers don't change.
 */
class TargetTracker {

    companion object {
        /** EMA weight on the previous position when a new detection arrives.
         *  0 = snap to detection (no smoothing); 1 = ignore detection (infinite lag). */
        private const val POS_EMA = 0.6f

        /** EMA weight on previous velocity. Higher = smoother but laggier velocity. */
        private const val VEL_EMA = 0.7f

        /** Multiplier applied to confidence on every miss. 0.85^12 ≈ 0.14. */
        private const val CONF_DECAY = 0.85f

        /** Below this confidence the tracker gives up and reports null. */
        private const val CONF_FLOOR = 0.15f

        /** Cap |vx|, |vy| (frame-fractions per tick) so a noisy detection at the
         *  screen edge can't fling the predicted box across the frame. */
        private const val MAX_VEL = 0.05f
    }

    private var x = 0f; private var y = 0f
    private var vx = 0f; private var vy = 0f
    private var conf = 0f

    /** Last seen box dimensions — held constant during prediction. */
    private var w = 0f; private var h = 0f
    private var area = 0f

    /** Last detection's metadata that we propagate through during predicted frames. */
    private var label = ""
    private var sourceW = 0; private var sourceH = 0

    private var hasLock = false

    /**
     * Called when the detector returned a real detection this frame.
     * Updates EMA position, refreshes velocity from the observed delta, resets confidence.
     */
    fun onHit(d: DetectorContract.DetectionResult) {
        if (!hasLock) {
            // First detection — anchor without smoothing, zero velocity.
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

    /**
     * Called when the detector returned null this frame.
     * Advances the predicted position by the current velocity and decays confidence.
     * Returns a predicted DetectionResult while confidence is still above CONF_FLOOR,
     * or null once the tracker has given up (caller should clear the overlay).
     */
    fun onMiss(): DetectorContract.DetectionResult? {
        if (!hasLock) return null

        x = (x + vx).coerceIn(0f, 1f)
        y = (y + vy).coerceIn(0f, 1f)
        conf *= CONF_DECAY

        if (conf < CONF_FLOOR) {
            reset()
            return null
        }

        // Damp velocity slightly during prediction — without a corrective observation,
        // we shouldn't keep accelerating in the assumed direction forever.
        vx *= 0.95f; vy *= 0.95f
        if (abs(vx) < 1e-4f) vx = 0f
        if (abs(vy) < 1e-4f) vy = 0f

        return current()
    }

    /** Snapshot of the tracker's current best estimate, formatted as a DetectionResult. */
    fun current(): DetectorContract.DetectionResult? {
        if (!hasLock) return null
        return DetectorContract.DetectionResult(
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
