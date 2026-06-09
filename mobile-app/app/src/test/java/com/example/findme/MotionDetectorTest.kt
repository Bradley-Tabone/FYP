package com.example.findme

import com.example.findme.MotionDetector.MotionState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionDetectorTest {

    @Test
    fun startsStationary() {
        val detector = MotionDetector { }
        assertEquals(MotionState.STATIONARY, detector.stateForTesting())
    }

    @Test
    fun smallMagnitudeStaysStationary() {
        val transitions = mutableListOf<MotionState>()
        val detector = MotionDetector { transitions.add(it) }

        feedSteady(detector, magnitude = 0.05f, startMs = 0L, durationMs = 3_000L)

        assertEquals(MotionState.STATIONARY, detector.stateForTesting())
        assertTrue(transitions.isEmpty())
    }

    @Test
    fun normalMotionEntersMovingImmediately() {
        val transitions = mutableListOf<MotionState>()
        val detector = MotionDetector { transitions.add(it) }

        feedSteady(detector, magnitude = 1.5f, startMs = 0L, durationMs = 500L)

        assertEquals(MotionState.MOVING, detector.stateForTesting())
        assertEquals(listOf(MotionState.MOVING), transitions)
    }

    @Test
    fun sustainedHighMagnitudeBecomesTooFast() {
        val transitions = mutableListOf<MotionState>()
        val detector = MotionDetector { transitions.add(it) }

        // 1.5 s of high acceleration — well past FAST_REQUIRED_MS (600 ms).
        feedSteady(detector, magnitude = 6.0f, startMs = 0L, durationMs = 1_500L)

        assertEquals(MotionState.MOVING_TOO_FAST, detector.stateForTesting())
        assertTrue(
            "expected eventual MOVING_TOO_FAST, got $transitions",
            transitions.contains(MotionState.MOVING_TOO_FAST)
        )
    }

    @Test
    fun briefSpikeDoesNotTriggerTooFast() {
        val transitions = mutableListOf<MotionState>()
        val detector = MotionDetector { transitions.add(it) }

        // Spike below FAST_REQUIRED_MS — only 200 ms of high motion, then back to calm.
        feedSteady(detector, magnitude = 6.0f, startMs = 0L, durationMs = 200L)
        feedSteady(detector, magnitude = 0.05f, startMs = 200L, durationMs = 3_000L)

        // The smoothed EMA may briefly cross MOVING during decay, but never TOO_FAST.
        assertTrue(
            "expected no MOVING_TOO_FAST, got $transitions",
            !transitions.contains(MotionState.MOVING_TOO_FAST)
        )
    }

    @Test
    fun stationaryRequiresSustainedCalmToReturn() {
        val transitions = mutableListOf<MotionState>()
        val detector = MotionDetector { transitions.add(it) }

        feedSteady(detector, magnitude = 1.5f, startMs = 0L, durationMs = 500L)
        assertEquals(MotionState.MOVING, detector.stateForTesting())

        // Two seconds of calm is below STILL_REQUIRED_MS (2.5 s) — must stay MOVING.
        feedSteady(detector, magnitude = 0.05f, startMs = 500L, durationMs = 2_000L)
        assertEquals(MotionState.MOVING, detector.stateForTesting())

        // Another second pushes total calm past 2.5 s — now drops to STATIONARY.
        feedSteady(detector, magnitude = 0.05f, startMs = 2_500L, durationMs = 1_000L)
        assertEquals(MotionState.STATIONARY, detector.stateForTesting())
    }

    @Test
    fun cooldownDelaysExitFromTooFast() {
        val transitions = mutableListOf<MotionState>()
        val detector = MotionDetector { transitions.add(it) }

        feedSteady(detector, magnitude = 6.0f, startMs = 0L, durationMs = 1_500L)
        assertEquals(MotionState.MOVING_TOO_FAST, detector.stateForTesting())

        // Drop just below the exit threshold; cooldown is 1.2 s.
        feedSteady(detector, magnitude = 1.0f, startMs = 1_500L, durationMs = 500L)
        assertEquals(MotionState.MOVING_TOO_FAST, detector.stateForTesting())

        // Past cooldown — must transition out.
        feedSteady(detector, magnitude = 1.0f, startMs = 2_000L, durationMs = 1_000L)
        assertEquals(MotionState.MOVING, detector.stateForTesting())
    }

    private fun feedSteady(
        detector: MotionDetector,
        magnitude: Float,
        startMs: Long,
        durationMs: Long,
        stepMs: Long = 60L
    ) {
        val end = startMs + durationMs
        var t = startMs
        while (t < end) {
            detector.feedForTesting(magnitude, t)
            t += stepMs
        }
        // Always sample the final timestamp so hold-time thresholds that land
        // between two step boundaries are still exercised.
        detector.feedForTesting(magnitude, end)
    }
}
