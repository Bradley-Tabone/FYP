package com.example.findme

/**
 * Provides spoken English guidance to direct a user toward a detected object.
 *
 * Guidance is delivered via the [speak] callback, wired to TTS in MainActivity.
 *
 * Behaviour:
 *  - On the first frame where a detection appears, announces "Possible <target> found."
 *  - While the object remains visible, periodically speaks direction (left/straight/right)
 *    and proximity (far/closer/very close) every [GUIDANCE_INTERVAL_MS] ms.
 *  - While the object is not visible, stays silent (the user already knows search is active
 *    from the start announcement).
 */
class AudioFeedbackManager(private val speak: (String) -> Unit) {

    data class DetectionResult(
        /** Horizontal centre of the bounding box: 0.0 = far left, 1.0 = far right. */
        val normalizedX: Float,
        /** Bounding-box area as a fraction of the total frame area: 0.0 = tiny, 1.0 = fills frame. */
        val normalizedArea: Float
    )

    @Volatile private var isRunning = false
    @Volatile private var currentResult: DetectionResult? = null
    @Volatile private var targetLabel: String = ""
    @Volatile private var pendingFoundAnnouncement = false

    private var guidanceThread: Thread? = null

    companion object {
        private const val GUIDANCE_INTERVAL_MS = 4000L
    }

    fun start(target: String) {
        targetLabel = target
        if (isRunning) return
        isRunning = true
        guidanceThread = Thread {
            while (isRunning) {
                if (pendingFoundAnnouncement) {
                    pendingFoundAnnouncement = false
                    speak("Possible $targetLabel found.")
                    try { Thread.sleep(GUIDANCE_INTERVAL_MS) } catch (e: InterruptedException) { break }
                    continue
                }

                val result = currentResult
                if (result != null) {
                    val direction = toDirection(result.normalizedX)
                    val proximity = toProximity(result.normalizedArea)
                    speak("$direction. $proximity.")
                } else {
                    speak("Scanning.")
                }

                try { Thread.sleep(GUIDANCE_INTERVAL_MS) } catch (e: InterruptedException) { break }
            }
        }.also { it.isDaemon = true; it.start() }
    }

    fun stop() {
        isRunning = false
        guidanceThread?.interrupt()
        guidanceThread = null
        currentResult = null
        pendingFoundAnnouncement = false
    }

    /** Call this from the detection callback to update direction/distance. Pass null when the
     *  target is not currently visible in the frame. */
    fun update(result: DetectionResult?) {
        if (result != null && currentResult == null) {
            pendingFoundAnnouncement = true
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
