package com.example.findme

/**
 * Provides spoken English guidance to direct a user toward a detected object.
 *
 * Guidance is delivered via the [speak] callback, wired to TTS in MainActivity.
 *
 * Behaviour:
 *  - On search start, after a short delay, instructs the user to slowly sweep the camera.
 *    The sweep reminder repeats every [SWEEP_REMINDER_INTERVAL_MS] ms until the first detection.
 *  - On the first frame where a detection appears, announces "Possible <target> found."
 *  - While the object remains visible, periodically speaks direction (left/straight/right)
 *    and proximity (far/closer/very close) every [GUIDANCE_INTERVAL_MS] ms.
 *  - While the object is not visible: speaks "Lost sight of X" / guides to last known position.
 */
class AudioFeedbackManager(private val speak: (String) -> Unit) {

    data class DetectionResult(
        /** Horizontal centre of the bounding box: 0.0 = far left, 1.0 = far right. */
        val normalizedX: Float,
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

    private var guidanceThread: Thread? = null

    companion object {
        private const val GUIDANCE_INTERVAL_MS        = 4000L
        private const val SWEEP_REMINDER_INTERVAL_MS  = 6000L
        private const val CONFIRMATION_THRESHOLD      = 0.80f
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
                    if (currentResult != null) {
                        speak("Possible $targetLabel found.")
                        try { Thread.sleep(GUIDANCE_INTERVAL_MS) } catch (e: InterruptedException) { break }
                    }
                    continue
                }

                val result = currentResult
                if (result != null) {
                    val direction = toDirection(result.normalizedX)
                    val proximity = toProximity(result.normalizedArea)
                    speak("$direction. $proximity.")
                    try { Thread.sleep(GUIDANCE_INTERVAL_MS) } catch (e: InterruptedException) { break }
                } else if (!everDetected) {
                    // Object not yet found — remind user to keep sweeping
                    speak("Keep sweeping slowly.")
                    try { Thread.sleep(SWEEP_REMINDER_INTERVAL_MS) } catch (e: InterruptedException) { break }
                } else {
                    // Object lost — speak feedback then wait
                    val last = lastConfirmedResult
                    if (isConfirmed && last != null) {
                        // High-confidence lock: guide toward last known position
                        val direction = toDirection(last.normalizedX)
                        speak("$targetLabel out of view. $direction.")
                    } else {
                        speak("Lost sight of $targetLabel. Keep searching.")
                    }
                    try { Thread.sleep(GUIDANCE_INTERVAL_MS) } catch (e: InterruptedException) { break }
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
    }

    /** Call this from the detection callback to update direction/distance. Pass null when the
     *  target is not currently visible in the frame. */
    fun update(result: DetectionResult?) {
        if (result != null) {
            if (currentResult == null) {
                // Announce whenever detection returns — first find or re-find after losing it
                pendingFoundAnnouncement = true
                if (!everDetected) everDetected = true
            }
            // Only a high-confidence detection can set or update the confirmed lock.
            // Lower-confidence detections still drive real-time guidance but cannot
            // overwrite a previously confirmed position.
            if (result.confidence >= CONFIRMATION_THRESHOLD) {
                isConfirmed = true
                lastConfirmedResult = result
            }
        }
        currentResult = result
    }

    // ── Guidance helpers ──────────────────────────────────────────────────────

    private fun toDirection(normalizedX: Float): String = when {
        normalizedX < 0.35f -> "Move left"
        normalizedX > 0.65f -> "Move right"
        else -> "Straight ahead"
    }

    private fun toProximity(normalizedArea: Float): String = when {
        normalizedArea < 0.05f -> "Keep moving forward"
        normalizedArea < 0.15f -> "Getting closer"
        normalizedArea < 0.30f -> "Almost there"
        else -> "Object is very close, stop"
    }
}
