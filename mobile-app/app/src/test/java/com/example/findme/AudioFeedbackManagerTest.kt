package com.example.findme

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI

class AudioFeedbackManagerTest {

    @Test
    fun noDetectionBeforeFirstResultIsSilent() {
        val manager = AudioFeedbackManager {}
        manager.resetForTesting("wallet")

        manager.update(null)

        assertNull(manager.currentGuidanceForTesting("wallet"))
    }

    @Test
    fun nullUpdateAfterDetectionIsSilent() {
        val manager = AudioFeedbackManager {}
        manager.resetForTesting("wallet")
        manager.update(detection(x = 0.50f, y = 0.50f, area = 0.14f))

        manager.update(null)

        assertNull(manager.currentGuidanceForTesting("wallet"))
    }

    @Test
    fun leftOfFrameUsesClockAndPanLeft() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.20f, y = 0.50f, area = 0.04f)
        )

        assertEquals("10 o'clock. Pan phone left slowly.", guidance)
    }

    @Test
    fun rightOfFrameUsesClockAndPanRight() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.80f, y = 0.50f, area = 0.04f)
        )

        assertEquals("2 o'clock. Pan phone right slowly.", guidance)
    }

    @Test
    fun centeredButSmallMovesCloser() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.50f, y = 0.50f, area = 0.04f)
        )

        assertEquals("Move forward slowly.", guidance)
    }

    @Test
    fun centeredAndLargeStopsAtObject() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.50f, y = 0.50f, area = 0.14f),
            centeredLargeFrames = 2
        )

        assertEquals("Stop. Wallet is in front of you.", guidance)
    }

    @Test
    fun arrivalRequiresTwoCenteredLargeFrames() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.50f, y = 0.50f, area = 0.14f),
            centeredLargeFrames = 1
        )

        assertEquals("Move forward slowly.", guidance)
    }

    @Test
    fun centeredLowWalletWithEnoughAreaUsesSurfaceReach() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.50f, y = 0.70f, area = 0.10f),
            centeredLargeFrames = 0
        )

        assertEquals("Wallet on surface in front of you. Reach ahead.", guidance)
    }

    @Test
    fun centeredLowKeysUseSmallerSurfaceAreaThanWallet() {
        val manager = AudioFeedbackManager {}
        val sameDetection = detection(x = 0.50f, y = 0.70f, area = 0.012f)

        assertEquals(
            "Keys on surface in front of you. Reach ahead.",
            manager.guidanceForTesting("keys", sameDetection, centeredLargeFrames = 0)
        )
        assertEquals(
            "Pan phone down slowly.",
            manager.guidanceForTesting("wallet", sameDetection, centeredLargeFrames = 0)
        )
    }

    @Test
    fun centeredLowWalletBelowSurfaceAreaMovesCloser() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.50f, y = 0.70f, area = 0.02f),
            centeredLargeFrames = 0
        )

        assertEquals("Pan phone down slowly.", guidance)
    }

    @Test
    fun offCenterLowDetectionStillPrioritizesPanGuidanceAndWarnsSurface() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.20f, y = 0.75f, area = 0.14f),
            centeredLargeFrames = 2
        )

        assertEquals("10 o'clock. Pan phone left and down slowly. Surface ahead.", guidance)
    }

    @Test
    fun centeredLowerFrameDetectionPansPhoneDownBeforeSurfaceThreshold() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.50f, y = 0.65f, area = 0.015f),
            centeredLargeFrames = 0
        )

        assertEquals("Pan phone down slowly.", guidance)
    }

    @Test
    fun lowFrameNonArrivedDetectionDoesNotSayAimLower() {
        val manager = AudioFeedbackManager {}

        val guidance = manager.guidanceForTesting(
            "wallet",
            detection(x = 0.50f, y = 0.90f, area = 0.015f),
            centeredLargeFrames = 0
        )

        assertEquals("Pan phone down slowly.", guidance)
        assertTrue(!guidance.contains("Aim lower"))
    }

    @Test
    fun keysUseSmallerArrivalAreaThanWallet() {
        val manager = AudioFeedbackManager {}
        val sameDetection = detection(x = 0.50f, y = 0.50f, area = 0.06f)

        assertEquals(
            "Stop. Keys are in front of you.",
            manager.guidanceForTesting("keys", sameDetection, centeredLargeFrames = 2)
        )
        assertEquals(
            "Move forward slowly.",
            manager.guidanceForTesting("wallet", sameDetection, centeredLargeFrames = 2)
        )
    }

    @Test
    fun smallCandidateLostDoesNotTriggerRejection() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")

        repeat(3) { frame ->
            manager.updateForTesting(detection(x = 0.50f, y = 0.50f, area = 0.04f), frame * 100L)
        }
        manager.updateForTesting(null, 300L)
        manager.tickForTesting(1_300L)

        assertTrue(spoken.isEmpty())
    }

    @Test
    fun oneFrameCloseCandidateLostDoesNotTriggerRejection() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")

        manager.updateForTesting(detection(x = 0.50f, y = 0.50f, area = 0.14f), 0L)
        manager.updateForTesting(null, 100L)
        manager.tickForTesting(1_100L)

        assertTrue(spoken.isEmpty())
    }

    @Test
    fun stableCloseCandidateLostForLessThanOneSecondStaysSilent() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")

        armCloseCandidate(manager, area = 0.14f)
        manager.updateForTesting(null, 300L)
        manager.tickForTesting(1_299L)

        assertTrue(spoken.isEmpty())
    }

    @Test
    fun stableCloseWalletCandidateLostForOneSecondTriggersRejection() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")

        armCloseCandidate(manager, area = 0.14f)
        manager.updateForTesting(null, 300L)
        manager.tickForTesting(1_300L)

        assertEquals(listOf("Sorry, that is not the wallet. Keep searching."), spoken)
    }

    @Test
    fun pluralTargetsUsePluralRejectionWording() {
        assertEquals(
            listOf("Sorry, those are not the keys. Keep searching."),
            rejectionCueFor(target = "keys", area = 0.06f)
        )
        assertEquals(
            listOf("Sorry, those are not the sunglasses. Keep searching."),
            rejectionCueFor(target = "sunglasses", area = 0.12f)
        )
    }

    @Test
    fun rejectionIsSpokenOnceForSameLostCandidate() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")

        armCloseCandidate(manager, area = 0.14f)
        manager.updateForTesting(null, 300L)
        manager.tickForTesting(1_300L)
        manager.tickForTesting(4_300L)

        assertEquals(listOf("Sorry, that is not the wallet. Keep searching."), spoken)
    }

    @Test
    fun newStableCandidateCanTriggerLaterRejection() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")

        armCloseCandidate(manager, area = 0.14f)
        manager.updateForTesting(null, 300L)
        manager.tickForTesting(1_300L)

        armCloseCandidate(manager, area = 0.14f, startAtMs = 2_000L)
        manager.updateForTesting(null, 2_300L)
        manager.tickForTesting(4_300L)

        assertEquals(
            listOf(
                "Sorry, that is not the wallet. Keep searching.",
                "Sorry, that is not the wallet. Keep searching."
            ),
            spoken
        )
    }

    @Test
    fun changedGuidanceInsideCooldownIsNotSpokenImmediately() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")

        manager.setCurrentResultForTesting(detection(x = 0.20f, y = 0.50f, area = 0.04f))
        manager.tickForTesting(0L)
        manager.setCurrentResultForTesting(detection(x = 0.80f, y = 0.50f, area = 0.04f))
        manager.tickForTesting(1_000L)

        assertEquals(listOf("10 o'clock. Pan phone left slowly."), spoken)
    }

    @Test
    fun changedGuidanceAfterCooldownIsSpoken() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")

        manager.setCurrentResultForTesting(detection(x = 0.20f, y = 0.50f, area = 0.04f))
        manager.tickForTesting(0L)
        manager.setCurrentResultForTesting(detection(x = 0.80f, y = 0.50f, area = 0.04f))
        manager.tickForTesting(1_800L)

        assertEquals(
            listOf(
                "10 o'clock. Pan phone left slowly.",
                "2 o'clock. Pan phone right slowly."
            ),
            spoken
        )
    }

    @Test
    fun repeatedGuidanceWaitsLongerThanChangedGuidance() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setCurrentResultForTesting(detection(x = 0.20f, y = 0.50f, area = 0.04f))

        manager.tickForTesting(0L)
        manager.tickForTesting(1_800L)
        manager.tickForTesting(3_000L)

        assertEquals(
            listOf(
                "10 o'clock. Pan phone left slowly.",
                "10 o'clock. Pan phone left slowly."
            ),
            spoken
        )
    }

    private fun detection(
        x: Float,
        y: Float,
        area: Float,
        confidence: Float = 0.90f
    ) = AudioFeedbackManager.DetectionResult(
        normalizedX = x,
        normalizedY = y,
        normalizedArea = area,
        confidence = confidence
    )

    private fun armCloseCandidate(
        manager: AudioFeedbackManager,
        area: Float,
        startAtMs: Long = 0L
    ) {
        manager.updateForTesting(detection(x = 0.50f, y = 0.50f, area = area), startAtMs)
        manager.updateForTesting(detection(x = 0.50f, y = 0.50f, area = area), startAtMs + 100L)
        manager.updateForTesting(detection(x = 0.50f, y = 0.50f, area = area), startAtMs + 200L)
    }

    private fun rejectionCueFor(target: String, area: Float): List<String> {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting(target)

        armCloseCandidate(manager, area = area)
        manager.updateForTesting(null, 300L)
        manager.tickForTesting(1_300L)

        return spoken
    }

    // ── Pace cues (driven by MotionDetector) ────────────────────────────

    @Test
    fun paceTooFastFiresWhenNoDetection() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.MOVING_TOO_FAST)

        manager.tickForTesting(2_000L)

        assertEquals(listOf("Slow down. Move the phone more steadily."), spoken)
    }

    @Test
    fun paceTooFastIsSuppressedWhileTargetVisible() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.MOVING_TOO_FAST)
        manager.setCurrentResultForTesting(detection(x = 0.20f, y = 0.50f, area = 0.04f))

        manager.tickForTesting(2_000L)

        assertEquals(listOf("10 o'clock. Pan phone left slowly."), spoken)
    }

    @Test
    fun paceTooSlowFiresAfterFiveSecondsStationaryWithNoDetection() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.STATIONARY)

        manager.tickForTesting(4_999L)
        assertTrue("must stay silent before 5s", spoken.isEmpty())

        manager.tickForTesting(5_001L)
        assertEquals(listOf("Try moving around to search for the wallet."), spoken)
    }

    @Test
    fun paceTooSlowDoesNotRepeatWithinTwelveSecondCooldown() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.STATIONARY)

        manager.tickForTesting(5_001L)   // first hint fires
        manager.tickForTesting(8_000L)   // 3 s later — must stay silent
        manager.tickForTesting(15_000L)  // 10 s later — still within cooldown
        manager.tickForTesting(17_001L)  // 12 s after the first hint — repeats

        assertEquals(
            listOf(
                "Try moving around to search for the wallet.",
                "Try moving around to search for the wallet."
            ),
            spoken
        )
    }

    @Test
    fun paceTooSlowIsSuppressedWhenMotionIsNormal() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.MOVING)

        manager.tickForTesting(10_000L)

        assertTrue("normal motion should not trigger pace cue", spoken.isEmpty())
    }

    @Test
    fun arrivalOverridesPaceTooFast() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.MOVING_TOO_FAST)
        manager.setCurrentResultForTesting(
            detection(x = 0.50f, y = 0.50f, area = 0.14f),
            centeredLargeFrames = 2
        )

        manager.tickForTesting(2_000L)

        assertEquals(listOf("Stop. Wallet is in front of you."), spoken)
    }

    // ── Bearing memory (driven by OrientationTracker) ────────────────────

    @Test
    fun bearingHintSpeaksTargetClockHourWhenNotVisible() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.STATIONARY)

        // Target was at the centre of the frame when stored, and the phone was
        // pointing at yaw=0. The user has since rotated -90° (left), so the
        // target is now at +90° relative → 3 o'clock.
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 0L)
        manager.updateOrientationForTesting((-PI / 2).toFloat())

        manager.tickForTesting(500L)

        assertEquals(listOf("Wallet at 3 o'clock."), spoken)
    }

    @Test
    fun bearingHintSuppressedWhileTargetVisible() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 0L)
        manager.updateOrientationForTesting((-PI / 2).toFloat())
        manager.setCurrentResultForTesting(detection(x = 0.20f, y = 0.50f, area = 0.04f))

        manager.tickForTesting(500L)

        assertEquals(listOf("10 o'clock. Pan phone left slowly."), spoken)
    }

    @Test
    fun bearingHintSuppressedWhenYawIsNaN() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(500L)
        manager.updateMotionForTesting(MotionDetector.MotionState.MOVING)
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 0L)
        // updateOrientation never called → currentYawRad stays NaN.

        manager.tickForTesting(500L)

        assertTrue("expected silence when no orientation data, got $spoken", spoken.isEmpty())
    }

    @Test
    fun bearingHintRespokenWhenClockHourChanges() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.STATIONARY)
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 0L)

        manager.updateOrientationForTesting((PI / 2).toFloat())  // target relative -90° → 9
        manager.tickForTesting(0L)

        manager.updateOrientationForTesting((PI / 3).toFloat())  // target relative -60° → 10
        manager.tickForTesting(1_800L)

        assertEquals(
            listOf("Wallet at 9 o'clock.", "Wallet at 10 o'clock."),
            spoken
        )
    }

    @Test
    fun bearingHintNotRespokenWhenClockHourUnchangedWithinRepeatWindow() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.STATIONARY)
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 0L)
        manager.updateOrientationForTesting((PI / 2).toFloat())

        manager.tickForTesting(0L)
        manager.tickForTesting(1_800L)

        assertEquals(listOf("Wallet at 9 o'clock."), spoken)
    }

    @Test
    fun bearingMemoryExpiresWhenStationary() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        // Keep "stationary hint" cue dormant by anchoring search-start close to "now".
        manager.setSearchStartedAtForTesting(8_000L)
        manager.updateMotionForTesting(MotionDetector.MotionState.STATIONARY)
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 0L)
        manager.updateOrientationForTesting((PI / 2).toFloat())

        manager.tickForTesting(8_001L)

        assertFalse(
            "expected no bearing hint after 8 s stationary expiry, got $spoken",
            spoken.any { it.contains("o'clock") }
        )
    }

    @Test
    fun bearingMemoryExpiresFasterWhenMovingTooFast() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.MOVING_TOO_FAST)
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 0L)
        manager.updateOrientationForTesting((PI / 2).toFloat())

        manager.tickForTesting(2_001L)

        assertFalse(
            "bearing must not fire after 2 s when motion is MOVING_TOO_FAST, got $spoken",
            spoken.any { it.contains("o'clock") }
        )
    }

    @Test
    fun arrivalOverridesBearingHint() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 0L)
        manager.updateOrientationForTesting((PI / 2).toFloat())
        manager.setCurrentResultForTesting(
            detection(x = 0.50f, y = 0.50f, area = 0.14f),
            centeredLargeFrames = 2
        )

        manager.tickForTesting(500L)

        assertEquals(listOf("Stop. Wallet is in front of you."), spoken)
    }

    @Test
    fun rejectionOverridesBearingHint() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        // Arm a close-candidate rejection; bearing memory is set in parallel.
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 0L)
        manager.updateOrientationForTesting((PI / 2).toFloat())

        armCloseCandidate(manager, area = 0.14f)
        manager.updateForTesting(null, 300L)
        manager.tickForTesting(1_300L)

        assertEquals(listOf("Sorry, that is not the wallet. Keep searching."), spoken)
    }

    @Test
    fun bearingHintWinsOverPaceTooFast() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)
        manager.updateMotionForTesting(MotionDetector.MotionState.MOVING_TOO_FAST)
        // Bearing memory just stored — comfortably within the 2 s MOVING_TOO_FAST lifetime.
        manager.setBearingMemoryForTesting(yawRad = 0f, frameX = 0.5f, atMs = 400L)
        manager.updateOrientationForTesting((PI / 2).toFloat())

        manager.tickForTesting(500L)

        assertEquals(listOf("Wallet at 9 o'clock."), spoken)
    }

    @Test
    fun rejectionOverridesPaceCue() {
        val spoken = mutableListOf<String>()
        val manager = AudioFeedbackManager { spoken.add(it) }
        manager.resetForTesting("wallet")
        manager.setSearchStartedAtForTesting(0L)

        armCloseCandidate(manager, area = 0.14f)
        manager.updateForTesting(null, 300L)
        manager.updateMotionForTesting(MotionDetector.MotionState.MOVING_TOO_FAST)
        manager.tickForTesting(1_300L)

        assertEquals(listOf("Sorry, that is not the wallet. Keep searching."), spoken)
    }
}
