package com.example.findme

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/*
 * Turns detector and sensor information into short spoken navigation messages.
 * It speaks from real detections, and only uses memory when the target was
 * seen recently and then leaves the camera view.
 */
class AudioFeedbackManager(private val speak: (String) -> Unit) {

    data class DetectionResult(
        // Horizontal box centre: 0.0 is left, 1.0 is right.
        val normalizedX: Float,
        // Vertical box centre: 0.0 is top, 1.0 is bottom.
        val normalizedY: Float,
        // Box area compared with the whole camera frame.
        val normalizedArea: Float,
        // Model confidence: 0.0 is unsure, 1.0 is very sure.
        val confidence: Float
    )

    internal enum class GuidancePhase {
        ALIGNING,
        APPROACHING,
        SURFACE,
        ARRIVED,
        REJECTED,
        PACE_TOO_FAST,
        PACE_TOO_SLOW,
        BEARING_HINT
    }

    /*
     * Last known phone direction and screen position from a real detection.
     * This helps give a clock direction if the target leaves the frame.
     */
    private data class BearingMemory(
        val storedYawRad: Float,
        val storedFrameX: Float,
        val storedAtMs: Long
    )

    internal data class GuidanceDecision(
        val phase: GuidancePhase,
        val message: String,
        // Clock-hour used only for bearing hints.
        val bearingHour: Int? = null
    )

    @Volatile private var isRunning = false
    @Volatile private var targetLabel: String = ""
    @Volatile private var currentResult: DetectionResult? = null
    @Volatile private var consecutiveCenteredLargeFrames = 0
    @Volatile private var consecutiveRejectableCloseFrames = 0
    @Volatile private var lastVisibleAtMs = 0L
    @Volatile private var rejectableCandidateArmed = false
    @Volatile private var rejectableCandidateLostAtMs = 0L
    @Volatile private var rejectionSpokenForLostCandidate = false

    /*
     * Latest movement state from MotionDetector.
     * The neutral default avoids pace warnings before sensor data arrives.
     */
    @Volatile private var motionState: MotionDetector.MotionState =
        MotionDetector.MotionState.MOVING
    @Volatile private var searchStartedAtMs = 0L

    /*
     * Last remembered target direction and latest phone yaw.
     * If yaw is NaN, orientation data is not ready yet.
     */
    @Volatile private var bearingMemory: BearingMemory? = null
    @Volatile private var currentYawRad: Float = Float.NaN
    private var lastBearingSpokenHour: Int = -1
    private var lastBearingSpokenAtMs: Long = 0L
    private var lastPaceSlowSpokenAtMs: Long = 0L

    /*
     * Trusted screen position for speech.
     * One-frame jumps are ignored, but repeated new positions are accepted.
     */
    @Volatile private var committedX = Float.NaN
    @Volatile private var committedY = Float.NaN
    @Volatile private var consecutiveFarFrames = 0

    private var lastSpoken = ""
    private var lastSpokenAtMs = 0L
    private var guidanceThread: Thread? = null

    companion object {
        private const val INITIAL_GUIDANCE_DELAY_MS = 1_500L
        private const val POLL_INTERVAL_MS = 200L
        private const val VISIBLE_GUIDANCE_INTERVAL_MS = 1_800L
        private const val REPEAT_SAME_COMMAND_AFTER_MS = 3_000L
        private const val MIN_SPEECH_GAP_MS = 1_400L

        private const val POSITION_JUMP_THRESHOLD = 0.30f
        private const val FAR_FRAMES_TO_SWITCH = 3
        private const val LOST_COMMIT_CLEAR_MS = 1_000L
        private const val REQUIRED_ARRIVAL_FRAMES = 2
        private const val REQUIRED_REJECTION_FRAMES = 3
        private const val REJECTION_LOST_MS = 1_000L

        private const val HIGH_FRAME_Y_THRESHOLD = 0.25f
        private const val LOW_FRAME_Y_THRESHOLD = 0.58f
        private const val SURFACE_Y_THRESHOLD = 0.58f

        /*
         * The camera does not know real distance from one image.
         * These are rough "close enough to reach" area guesses for each object.
         */
        private const val WALLET_SURFACE_AREA = 0.025f
        private const val SUNGLASSES_SURFACE_AREA = 0.018f
        private const val KEYS_SURFACE_AREA = 0.008f
        private const val DEFAULT_SURFACE_AREA = 0.018f

        // Wait this long before suggesting movement when the target has not been seen.
        private const val STATIONARY_HINT_AFTER_MS = 5_000L

        // Wait longer before repeating the same stationary hint.
        private const val STATIONARY_HINT_REPEAT_MS = 12_000L

        /*
         * Approximate horizontal camera view.
         * Bearing maths uses this to turn frame position into a left/right angle.
         */
        private const val HORIZONTAL_FOV_DEG = 70f
        private const val HORIZONTAL_FOV_RAD =
            (HORIZONTAL_FOV_DEG * PI / 180.0).toFloat()

        /*
         * How long to trust a remembered bearing.
         * Faster movement makes the memory expire sooner.
         */
        private const val BEARING_LIFETIME_STATIONARY_MS = 8_000L
        private const val BEARING_LIFETIME_MOVING_MS = 4_000L
        private const val BEARING_LIFETIME_TOO_FAST_MS = 2_000L

        /*
         * Repeat a bearing when the clock hour changes, or after this much silence.
         */
        private const val BEARING_REPEAT_MS = 2_500L
    }

    fun start(target: String) {
        resetState(target)
        searchStartedAtMs = System.currentTimeMillis()
        if (isRunning) return

        isRunning = true
        guidanceThread = Thread {
            /*
             * Give MainActivity's "Searching for ..." message time to finish
             * before detector guidance starts speaking.
             */
            sleepOrStop(INITIAL_GUIDANCE_DELAY_MS)
            while (isRunning) {
                speakIfDue(System.currentTimeMillis())
                sleepOrStop(POLL_INTERVAL_MS)
            }
        }.also {
            it.isDaemon = true
            it.start()
        }
    }

    fun stop() {
        isRunning = false
        guidanceThread?.interrupt()
        guidanceThread = null
        resetState("")
    }

    /*
     * Receives real detector updates.
     * Pass null when the target is not visible.
     */
    fun update(result: DetectionResult?) {
        update(result, System.currentTimeMillis())
    }

    /*
     * Stores the latest movement state.
     * The speech loop reads this when deciding pace guidance.
     */
    fun updateMotion(state: MotionDetector.MotionState) {
        motionState = state
    }

    /*
     * Stores the latest yaw reading.
     * NaN means orientation guidance is unavailable.
     */
    fun updateOrientation(yawRadians: Float) {
        currentYawRad = yawRadians
    }

    private fun update(result: DetectionResult?, now: Long) {
        if (result == null) {
            currentResult = null
            consecutiveFarFrames = 0
            consecutiveCenteredLargeFrames = 0
            consecutiveRejectableCloseFrames = 0
            if (rejectableCandidateArmed && rejectableCandidateLostAtMs == 0L) {
                rejectableCandidateLostAtMs = now
            }
            if (lastVisibleAtMs != 0L && now - lastVisibleAtMs > LOST_COMMIT_CLEAR_MS) {
                committedX = Float.NaN
                committedY = Float.NaN
            }
            return
        }

        val accepted = stabilized(result) ?: return
        currentResult = accepted
        lastVisibleAtMs = now
        rejectableCandidateLostAtMs = 0L

        /*
         * Refresh bearing memory on every real detection.
         * This memory is only used if the target later leaves the frame.
         */
        val yaw = currentYawRad
        if (!yaw.isNaN()) {
            bearingMemory = BearingMemory(yaw, accepted.normalizedX, now)
            lastBearingSpokenHour = -1
        }

        consecutiveCenteredLargeFrames = if (isCentered(accepted) && isLargeEnough(targetLabel, accepted)) {
            consecutiveCenteredLargeFrames + 1
        } else {
            0
        }

        if (isLargeEnough(targetLabel, accepted)) {
            consecutiveRejectableCloseFrames++
            if (consecutiveRejectableCloseFrames >= REQUIRED_REJECTION_FRAMES) {
                rejectableCandidateArmed = true
                rejectionSpokenForLostCandidate = false
            }
        } else {
            consecutiveRejectableCloseFrames = 0
            rejectableCandidateArmed = false
            rejectionSpokenForLostCandidate = false
        }
    }

    private fun speakIfDue(now: Long) {
        val decision = currentGuidanceDecision(now) ?: return
        val message = decision.message
        val elapsed = now - lastSpokenAtMs

        if (lastSpoken.isNotEmpty() && elapsed < MIN_SPEECH_GAP_MS) return

        val requiredInterval = if (message == lastSpoken) {
            REPEAT_SAME_COMMAND_AFTER_MS
        } else {
            VISIBLE_GUIDANCE_INTERVAL_MS
        }

        if (lastSpoken.isNotEmpty() && elapsed < requiredInterval) return

        speak(message)
        lastSpoken = message
        lastSpokenAtMs = now
        if (decision.phase == GuidancePhase.REJECTED) {
            markRejectionSpoken()
        }
        if (decision.phase == GuidancePhase.BEARING_HINT && decision.bearingHour != null) {
            lastBearingSpokenHour = decision.bearingHour
            lastBearingSpokenAtMs = now
        }
        if (decision.phase == GuidancePhase.PACE_TOO_SLOW) {
            lastPaceSlowSpokenAtMs = now
        }
    }

    private fun currentGuidanceDecision(now: Long): GuidanceDecision? {
        val result = currentResult
        if (result != null) {
            return decideVisibleGuidance(targetLabel, result, consecutiveCenteredLargeFrames)
        }
        val rejection = decideLostCandidateGuidance(now)
        if (rejection != null) return rejection
        val bearing = decideBearingHint(now)
        if (bearing != null) return bearing
        return decidePaceGuidance(now)
    }

    private fun decideBearingHint(now: Long): GuidanceDecision? {
        val mem = bearingMemory ?: return null
        val yaw = currentYawRad
        if (yaw.isNaN()) return null
        if (now - mem.storedAtMs > bearingMemoryLifetimeMs()) {
            bearingMemory = null
            lastBearingSpokenHour = -1
            return null
        }
        val targetWorldYaw = mem.storedYawRad + (mem.storedFrameX - 0.5f) * HORIZONTAL_FOV_RAD
        val relRad = wrapToPi(targetWorldYaw - yaw)
        val hour = clockHourFromRadians(relRad)
        val sameHour = hour == lastBearingSpokenHour
        val elapsed = now - lastBearingSpokenAtMs
        if (sameHour && elapsed < BEARING_REPEAT_MS) return null
        return GuidanceDecision(
            phase = GuidancePhase.BEARING_HINT,
            message = bearingMessage(targetLabel, hour),
            bearingHour = hour
        )
    }

    private fun bearingMemoryLifetimeMs(): Long = when (motionState) {
        MotionDetector.MotionState.STATIONARY -> BEARING_LIFETIME_STATIONARY_MS
        MotionDetector.MotionState.MOVING -> BEARING_LIFETIME_MOVING_MS
        MotionDetector.MotionState.MOVING_TOO_FAST -> BEARING_LIFETIME_TOO_FAST_MS
    }

    private fun bearingMessage(target: String, hour: Int): String {
        val word = target.lowercase()
            .replaceFirstChar { it.uppercase() }
            .ifBlank { "Object" }
        return "$word at $hour o'clock."
    }

    private fun wrapToPi(rad: Float): Float {
        val twoPi = (2.0 * PI).toFloat()
        var r = rad
        while (r > PI) r -= twoPi
        while (r < -PI) r += twoPi
        return r
    }

    /*
     * Converts an angle around the user into a clock hour.
     * 0 radians means straight ahead, and positive angles move to the right.
     */
    private fun clockHourFromRadians(rad: Float): Int {
        // Each clock hour represents 30 degrees.
        val degrees = (rad * 180.0 / PI).toFloat()
        val rawHour = (degrees / 30f).roundToInt()
        // Move the range into 1..12, including negative angles.
        return ((rawHour + 11) % 12) + 1
    }

    private fun decidePaceGuidance(now: Long): GuidanceDecision? {
        if (targetLabel.isEmpty()) return null
        return when (motionState) {
            MotionDetector.MotionState.MOVING_TOO_FAST ->
                GuidanceDecision(GuidancePhase.PACE_TOO_FAST, paceTooFastMessage())
            MotionDetector.MotionState.STATIONARY -> {
                val sinceVisible = if (lastVisibleAtMs == 0L) {
                    now - searchStartedAtMs
                } else {
                    now - lastVisibleAtMs
                }
                val sinceLastHint = now - lastPaceSlowSpokenAtMs
                val cooldownExpired =
                    lastPaceSlowSpokenAtMs == 0L || sinceLastHint >= STATIONARY_HINT_REPEAT_MS
                if (sinceVisible >= STATIONARY_HINT_AFTER_MS && cooldownExpired) {
                    GuidanceDecision(GuidancePhase.PACE_TOO_SLOW, paceTooSlowMessage(targetLabel))
                } else {
                    null
                }
            }
            MotionDetector.MotionState.MOVING -> null
        }
    }

    private fun paceTooFastMessage(): String =
        "Slow down. Move the phone more steadily."

    private fun paceTooSlowMessage(target: String): String {
        val lower = target.lowercase()
        return if (lower.isBlank()) {
            "Try moving around to search."
        } else {
            "Try moving around to search for the $lower."
        }
    }

    private fun stabilized(result: DetectionResult): DetectionResult? {
        if (committedX.isNaN()) {
            committedX = result.normalizedX
            committedY = result.normalizedY
            consecutiveFarFrames = 0
            return result
        }

        val dist = sqrt(
            (result.normalizedX - committedX).pow(2) +
                (result.normalizedY - committedY).pow(2)
        )

        if (dist > POSITION_JUMP_THRESHOLD) {
            consecutiveFarFrames++
            if (consecutiveFarFrames < FAR_FRAMES_TO_SWITCH) return null

            committedX = result.normalizedX
            committedY = result.normalizedY
            consecutiveFarFrames = 0
            return result
        }

        consecutiveFarFrames = 0
        committedX = committedX * 0.8f + result.normalizedX * 0.2f
        committedY = committedY * 0.8f + result.normalizedY * 0.2f
        return result.copy(normalizedX = committedX, normalizedY = committedY)
    }

    private fun sleepOrStop(ms: Long) {
        try {
            Thread.sleep(ms)
        } catch (e: InterruptedException) {
            isRunning = false
        }
    }

    private fun resetState(target: String) {
        targetLabel = target
        currentResult = null
        consecutiveCenteredLargeFrames = 0
        consecutiveRejectableCloseFrames = 0
        lastVisibleAtMs = 0L
        rejectableCandidateArmed = false
        rejectableCandidateLostAtMs = 0L
        rejectionSpokenForLostCandidate = false
        committedX = Float.NaN
        committedY = Float.NaN
        consecutiveFarFrames = 0
        lastSpoken = ""
        lastSpokenAtMs = 0L
        motionState = MotionDetector.MotionState.MOVING
        searchStartedAtMs = 0L
        bearingMemory = null
        lastBearingSpokenHour = -1
        lastBearingSpokenAtMs = 0L
        lastPaceSlowSpokenAtMs = 0L
        /*
         * currentYawRad is a live sensor reading, not search state.
         * Bearing memory is cleared, so old yaw alone cannot create guidance.
         */
    }

    // Testable guidance helpers used without starting the speech thread.

    internal fun decideVisibleGuidance(
        target: String,
        result: DetectionResult,
        centeredLargeFrames: Int = REQUIRED_ARRIVAL_FRAMES
    ): GuidanceDecision {
        val centered = isCentered(result)

        if (!centered) {
            return GuidanceDecision(GuidancePhase.ALIGNING, alignmentMessage(target, result))
        }

        if (isLikelySurfaceReach(target, result)) {
            return GuidanceDecision(GuidancePhase.SURFACE, surfaceReachMessage(target))
        }

        val arrived = isLargeEnough(target, result) && centeredLargeFrames >= REQUIRED_ARRIVAL_FRAMES
        if (arrived) {
            return GuidanceDecision(GuidancePhase.ARRIVED, arrivedMessage(target))
        }

        return GuidanceDecision(GuidancePhase.APPROACHING, centeredApproachMessage(result))
    }

    internal fun guidanceForTesting(
        target: String,
        result: DetectionResult,
        centeredLargeFrames: Int = REQUIRED_ARRIVAL_FRAMES
    ): String = decideVisibleGuidance(target, result, centeredLargeFrames).message

    internal fun currentGuidanceForTesting(
        target: String,
        now: Long = System.currentTimeMillis()
    ): String? {
        targetLabel = target
        return currentGuidanceDecision(now)?.message
    }

    internal fun resetForTesting(target: String) {
        resetState(target)
    }

    internal fun setCurrentResultForTesting(
        result: DetectionResult?,
        centeredLargeFrames: Int = 0
    ) {
        currentResult = result
        consecutiveCenteredLargeFrames = centeredLargeFrames
    }

    internal fun tickForTesting(now: Long) {
        speakIfDue(now)
    }

    internal fun updateForTesting(result: DetectionResult?, now: Long) {
        update(result, now)
    }

    internal fun updateMotionForTesting(state: MotionDetector.MotionState) {
        motionState = state
    }

    internal fun setSearchStartedAtForTesting(now: Long) {
        searchStartedAtMs = now
    }

    internal fun updateOrientationForTesting(yawRad: Float) {
        currentYawRad = yawRad
    }

    internal fun setBearingMemoryForTesting(yawRad: Float, frameX: Float, atMs: Long) {
        bearingMemory = BearingMemory(yawRad, frameX, atMs)
        lastBearingSpokenHour = -1
        lastBearingSpokenAtMs = 0L
    }

    private fun alignmentMessage(target: String, result: DetectionResult): String {
        // This is only called when the target is clearly left or right of centre.
        val direction = if (result.normalizedX < 0.35f) "left" else "right"
        val horizontal = horizontalPanMessage(result, direction)
        return appendSurfaceHint(target, horizontal, result)
    }

    private fun centeredApproachMessage(result: DetectionResult): String {
        /*
         * The target is already roughly centred.
         * Give one next action instead of saying "centred" out loud.
         */
        return when {
            result.normalizedY < HIGH_FRAME_Y_THRESHOLD -> "Pan phone up slowly."
            result.normalizedY > LOW_FRAME_Y_THRESHOLD -> "Pan phone down slowly."
            else -> "Move forward slowly."
        }
    }

    private fun arrivedMessage(target: String): String {
        val lowerTarget = target.lowercase()
        val displayTarget = lowerTarget.replaceFirstChar { it.uppercase() }
        return when {
            displayTarget.isBlank() -> "Stop. Object is in front of you."
            lowerTarget == "keys" || lowerTarget == "sunglasses" ->
                "Stop. $displayTarget are in front of you."
            else -> "Stop. $displayTarget is in front of you."
        }
    }

    private fun decideLostCandidateGuidance(now: Long): GuidanceDecision? {
        if (!rejectableCandidateArmed) return null
        if (rejectionSpokenForLostCandidate) return null
        if (rejectableCandidateLostAtMs == 0L) return null
        if (now - rejectableCandidateLostAtMs < REJECTION_LOST_MS) return null
        return GuidanceDecision(GuidancePhase.REJECTED, rejectionMessage(targetLabel))
    }

    private fun markRejectionSpoken() {
        rejectionSpokenForLostCandidate = true
        rejectableCandidateArmed = false
        rejectableCandidateLostAtMs = 0L
        consecutiveRejectableCloseFrames = 0
    }

    private fun rejectionMessage(target: String): String {
        return when (target.lowercase()) {
            "keys" -> "Sorry, those are not the keys. Keep searching."
            "sunglasses" -> "Sorry, those are not the sunglasses. Keep searching."
            "wallet" -> "Sorry, that is not the wallet. Keep searching."
            else -> "Sorry, that is not the object. Keep searching."
        }
    }

    private fun isLikelySurfaceReach(target: String, result: DetectionResult): Boolean {
        return isCentered(result) &&
            result.normalizedY >= SURFACE_Y_THRESHOLD &&
            result.normalizedArea >= surfaceAreaThreshold(target)
    }

    private fun surfaceReachMessage(target: String): String {
        val displayTarget = target.lowercase().replaceFirstChar { it.uppercase() }
        val targetText = displayTarget.ifBlank { "Object" }
        return "$targetText on surface in front of you. Reach ahead."
    }

    private fun horizontalPanMessage(result: DetectionResult, direction: String): String {
        val vertical = when {
            result.normalizedY < HIGH_FRAME_Y_THRESHOLD -> " and up"
            result.normalizedY > LOW_FRAME_Y_THRESHOLD -> " and down"
            else -> ""
        }
        return "${toClockHour(result.normalizedX)}. Pan phone $direction$vertical slowly."
    }

    private fun appendSurfaceHint(target: String, message: String, result: DetectionResult): String {
        return if (isLikelySurfaceProximity(target, result)) "$message Surface ahead." else message
    }

    private fun toClockHour(normalizedX: Float): String {
        if (abs(normalizedX - 0.5f) < 0.05f) return "12 o'clock"
        val rawHour = (normalizedX * 6f + 9f).roundToInt()
        val hour = (rawHour % 12).let { if (it == 0) 12 else it }
        return "$hour o'clock"
    }

    private fun isCentered(result: DetectionResult): Boolean {
        /*
         * Treat the middle 30 percent of the frame as centred.
         * This avoids noisy tiny left/right corrections.
         */
        return result.normalizedX in 0.35f..0.65f
    }

    private fun isLargeEnough(target: String, result: DetectionResult): Boolean {
        return result.normalizedArea >= arrivalAreaThreshold(target)
    }

    private fun isLikelySurfaceProximity(target: String, result: DetectionResult): Boolean {
        return result.normalizedY >= SURFACE_Y_THRESHOLD &&
            result.normalizedArea >= surfaceAreaThreshold(target)
    }

    private fun arrivalAreaThreshold(target: String): Float {
        return when (target.lowercase()) {
            "keys" -> 0.035f
            "sunglasses" -> 0.065f
            "wallet" -> 0.075f
            else -> 0.065f
        }
    }

    private fun surfaceAreaThreshold(target: String): Float {
        return when (target.lowercase()) {
            "keys" -> KEYS_SURFACE_AREA
            "sunglasses" -> SUNGLASSES_SURFACE_AREA
            "wallet" -> WALLET_SURFACE_AREA
            else -> DEFAULT_SURFACE_AREA
        }
    }
}
