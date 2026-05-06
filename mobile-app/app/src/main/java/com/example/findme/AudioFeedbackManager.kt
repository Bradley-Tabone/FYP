package com.example.findme

import kotlin.math.*

/**
 * Provides spoken English guidance to direct a blind user toward a detected object.
 *
 * Guidance is delivered via the [speak] callback, wired to TTS in MainActivity.
 *
 * Direction model — **horizontal clock + vertical cue**:
 *   The camera only sees forward, so clock hours are limited to the forward arc
 *   (9 through 12 through 3). normalizedX maps linearly to this range.
 *   When the object is near the frame centre, guidance says "directly ahead".
 *
 *   Vertical position (normalizedY) drives separate cues: "above you",
 *   "below you", "on the surface to your left/right", "reach down".
 *   These are more useful than clock hours 4-7 (which would wrongly imply
 *   "behind you" to the user).
 *
 * Proximity — **arm's-reach aware**:
 *   Thresholds are tuned so "within arm's reach" fires at a comfortable
 *   reaching distance (~60 cm), not when the phone is touching the object.
 *
 * Detection commitment:
 *   Once a high-confidence detection is confirmed, the manager *commits* to
 *   that position. If a new detection appears far from the committed spot
 *   (likely a false positive or a different object), it is ignored for up to
 *   [FAR_FRAMES_TO_SWITCH] consecutive frames. This prevents the user from
 *   being pulled away from the real object by a momentary false positive
 *   during a walk-by. If the far detection persists, it is accepted as a
 *   genuine new find.
 */
class AudioFeedbackManager(private val speak: (String) -> Unit) {

    data class DetectionResult(
        /** Horizontal centre of the bounding box: 0.0 = far left, 1.0 = far right. */
        val normalizedX: Float,
        /** Vertical centre of the bounding box: 0.0 = top edge, 1.0 = bottom edge. */
        val normalizedY: Float,
        /** Bounding-box area as a fraction of the total frame area: 0.0 = tiny, 1.0 = fills frame. */
        val normalizedArea: Float,
        /** Detection confidence: 0.0 = uncertain, 1.0 = certain. */
        val confidence: Float
    )

    @Volatile private var isRunning = false
    @Volatile private var currentResult: DetectionResult? = null
    @Volatile private var targetLabel: String = ""
    @Volatile private var pendingFoundAnnouncement = false
    @Volatile private var everDetected = false
    @Volatile private var isConfirmed = false
    @Volatile private var lastConfirmedResult: DetectionResult? = null
    @Volatile private var lostSightAnnounced = false

    // Detection commitment — prevents jumping to distant false positives
    @Volatile private var committedX = Float.NaN
    @Volatile private var committedY = Float.NaN
    @Volatile private var consecutiveFarFrames = 0

    private var guidanceThread: Thread? = null

    companion object {
        private const val GUIDANCE_INTERVAL_MS        = 4000L
        private const val SWEEP_REMINDER_INTERVAL_MS  = 6000L
        private const val CONFIRMATION_THRESHOLD      = 0.80f

        /** How far (normalised Euclidean distance) a new detection can be from the
         *  committed position before it is considered a "jump". */
        private const val POSITION_JUMP_THRESHOLD     = 0.30f
        /** How many consecutive far-away detections are needed before the commitment
         *  switches to the new position. Filters brief false positives. */
        private const val FAR_FRAMES_TO_SWITCH        = 10
    }

    fun start(target: String) {
        targetLabel  = target
        everDetected = false
        if (isRunning) return
        isRunning = true
        guidanceThread = Thread {
            // Initial sweep instruction — give the "Searching for X" TTS time to finish first
            try { Thread.sleep(2000) } catch (e: InterruptedException) { return@Thread }
            if (isRunning && !everDetected) {
                speak("Slowly sweep the camera around the room.")
            }

            while (isRunning) {
                if (pendingFoundAnnouncement) {
                    pendingFoundAnnouncement = false
                    val snap = currentResult
                    if (snap != null) {
                        speak("Possible $targetLabel found.")
                        try { Thread.sleep(GUIDANCE_INTERVAL_MS) } catch (e: InterruptedException) { break }
                    }
                    continue
                }

                val result = currentResult
                when {
                    result != null -> {
                        speak(composeGuidance(result))
                        try { Thread.sleep(GUIDANCE_INTERVAL_MS) } catch (e: InterruptedException) { break }
                    }
                    !everDetected -> {
                        // Sweep instruction already given once at start — wait silently.
                        try { Thread.sleep(SWEEP_REMINDER_INTERVAL_MS) } catch (e: InterruptedException) { break }
                    }
                    !lostSightAnnounced -> {
                        // Object just left the frame — announce once, then stay silent.
                        lostSightAnnounced = true
                        val last = lastConfirmedResult
                        if (isConfirmed && last != null) speak(composeOutOfViewGuidance(last))
                        else speak("Lost sight of $targetLabel. Keep searching.")
                        try { Thread.sleep(GUIDANCE_INTERVAL_MS) } catch (e: InterruptedException) { break }
                    }
                    else -> {
                        // Already announced the loss — wait silently for a re-detection.
                        try { Thread.sleep(GUIDANCE_INTERVAL_MS) } catch (e: InterruptedException) { break }
                    }
                }
            }
        }.also { it.isDaemon = true; it.start() }
    }

    fun stop() {
        isRunning = false
        guidanceThread?.interrupt()
        guidanceThread = null
        currentResult = null
        pendingFoundAnnouncement = false
        everDetected = false
        isConfirmed = false
        lastConfirmedResult = null
        lostSightAnnounced = false
        committedX = Float.NaN
        committedY = Float.NaN
        consecutiveFarFrames = 0
    }

    /**
     * Call from the detection callback every frame. Pass null when the target
     * is not visible. Applies commitment filtering — a detection far from the
     * committed position is suppressed unless it persists.
     */
    fun update(result: DetectionResult?) {
        if (result != null) {
            // ── Commitment filtering ─────────────────────────────────────
            if (!committedX.isNaN()) {
                val dist = sqrt(
                    (result.normalizedX - committedX).pow(2) +
                    (result.normalizedY - committedY).pow(2)
                )
                if (dist > POSITION_JUMP_THRESHOLD) {
                    consecutiveFarFrames++
                    if (consecutiveFarFrames < FAR_FRAMES_TO_SWITCH) {
                        // Suppress this detection — keep guiding to committed position
                        return
                    }
                    // Enough consecutive far frames — accept the new position
                    committedX = result.normalizedX
                    committedY = result.normalizedY
                    consecutiveFarFrames = 0
                } else {
                    // Near committed position — smooth-update the commitment
                    consecutiveFarFrames = 0
                    committedX = committedX * 0.8f + result.normalizedX * 0.2f
                    committedY = committedY * 0.8f + result.normalizedY * 0.2f
                }
            }

            // ── Normal update logic ──────────────────────────────────────
            if (currentResult == null) {
                pendingFoundAnnouncement = true
                if (!everDetected) everDetected = true
                lostSightAnnounced = false  // Object reappeared — allow lost-sight to fire again next time
            }
            if (result.confidence >= CONFIRMATION_THRESHOLD) {
                isConfirmed = true
                lastConfirmedResult = result
                // Set initial commitment on first confirmed detection
                if (committedX.isNaN()) {
                    committedX = result.normalizedX
                    committedY = result.normalizedY
                }
            }
        } else {
            consecutiveFarFrames = 0
        }
        currentResult = result
    }

    // ── Guidance helpers ──────────────────────────────────────────────────────

    /**
     * Map horizontal position to a clock hour on the **forward arc only** (9→12→3).
     * normalizedX 0.0 = 9 o'clock (far left), 0.5 = 12 (ahead), 1.0 = 3 o'clock (far right).
     * Returns "directly ahead" when the object is near the horizontal centre.
     */
    private fun toClockHour(normalizedX: Float): String {
        if (abs(normalizedX - 0.5f) < 0.15f) return "directly ahead"
        // Linear map: 0→9, 0.5→12, 1→3  (i.e. hours = normalizedX * 6 + 9, mod 12)
        val rawHour = (normalizedX * 6f + 9f).roundToInt()
        val hour = (rawHour % 12).let { if (it == 0) 12 else it }
        return "$hour o'clock"
    }

    private fun toProximity(normalizedArea: Float): String = when {
        normalizedArea < 0.03f -> "keep walking forward"
        normalizedArea < 0.08f -> "getting closer"
        normalizedArea < 0.15f -> "within arm's reach"
        else                   -> "right in front of you"
    }

    /**
     * Vertical/surface cue based on where the object sits in the frame.
     *
     * Lower-frame detections mean the object is on a surface below the camera —
     * the user needs "reach down" / "on the surface" guidance, not a clock hour.
     * Returns null when no special vertical cue is needed.
     */
    private fun toVerticalCue(normalizedX: Float, normalizedY: Float, normalizedArea: Float): String? {
        if (normalizedY < 0.15f) return "above you"

        // Object near the very bottom of frame — urgent, about to leave view
        if (normalizedY > 0.85f) return "right below you, point your phone down"

        // Object in lower portion and within reach — give surface-level guidance
        if (normalizedY > 0.70f && normalizedArea >= 0.08f) {
            val side = when {
                normalizedX < 0.35f -> "to your left"
                normalizedX > 0.65f -> "to your right"
                else                -> "right in front of you"
            }
            return "on the surface $side"
        }

        // Object in lower third but still far away — gentle hint
        if (normalizedY > 0.70f) return "below you"

        return null
    }

    /** Compose a natural-language guidance sentence for an active detection. */
    private fun composeGuidance(result: DetectionResult): String {
        val clock       = toClockHour(result.normalizedX)
        val proximity   = toProximity(result.normalizedArea)
        val verticalCue = toVerticalCue(result.normalizedX, result.normalizedY, result.normalizedArea)

        // When the object is in the lower part of the frame and close, the vertical/surface
        // cue is more useful than the clock direction — lead with it.
        if (verticalCue != null && result.normalizedY > 0.70f && result.normalizedArea >= 0.08f) {
            return "$targetLabel $verticalCue. ${proximity.replaceFirstChar { it.uppercase() }}."
        }

        val directionPhrase = if (clock == "directly ahead") "$targetLabel directly ahead"
                              else "$targetLabel at $clock"

        val locationPart = if (verticalCue != null) "$directionPhrase, $verticalCue"
                           else directionPhrase

        return "$locationPart. ${proximity.replaceFirstChar { it.uppercase() }}."
    }

    /** Compose guidance when the object was recently visible but is now out of frame. */
    private fun composeOutOfViewGuidance(lastResult: DetectionResult): String {
        val clock       = toClockHour(lastResult.normalizedX)
        val verticalCue = toVerticalCue(lastResult.normalizedX, lastResult.normalizedY, lastResult.normalizedArea)

        // If the object was below/on surface when lost, prioritise that info
        if (verticalCue != null && lastResult.normalizedY > 0.70f) {
            return "$targetLabel lost. It was $verticalCue."
        }

        return if (clock == "directly ahead") {
            if (verticalCue != null) "$targetLabel lost, was $verticalCue."
            else "$targetLabel lost. It was straight ahead."
        } else {
            if (verticalCue != null) "$targetLabel lost. Try $clock, $verticalCue."
            else "$targetLabel lost. Try $clock."
        }
    }
}
