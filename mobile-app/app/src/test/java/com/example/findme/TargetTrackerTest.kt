package com.example.findme

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TargetTrackerTest {

    @Test
    fun firstHitBecomesCurrentDetection() {
        val tracker = TargetTracker()
        val detection = detection(x = 0.25f, y = 0.40f, confidence = 0.90f)

        tracker.onHit(detection)
        val current = tracker.current()

        assertNotNull(current)
        assertEquals(0.25f, current!!.normalizedX, 0.001f)
        assertEquals(0.40f, current.normalizedY, 0.001f)
        assertEquals(0.90f, current.confidence, 0.001f)
    }

    @Test
    fun missPredictsBrieflyThenExpires() {
        val tracker = TargetTracker()
        tracker.onHit(detection(x = 0.50f, y = 0.50f, confidence = 0.90f))

        val firstMiss = tracker.onMiss()
        assertNotNull(firstMiss)
        assertTrue(firstMiss!!.confidence < 0.90f)

        var current: DetectorTypes.DetectionResult? = firstMiss
        repeat(20) { current = tracker.onMiss() }

        assertNull(current)
        assertNull(tracker.current())
    }

    @Test
    fun secondHitSmoothsTowardNewPosition() {
        val tracker = TargetTracker()
        tracker.onHit(detection(x = 0.20f, y = 0.20f, confidence = 0.90f))
        tracker.onHit(detection(x = 0.40f, y = 0.40f, confidence = 0.90f))

        val current = tracker.current()

        assertNotNull(current)
        assertTrue(current!!.normalizedX > 0.20f)
        assertTrue(current.normalizedX < 0.40f)
        assertTrue(current.normalizedY > 0.20f)
        assertTrue(current.normalizedY < 0.40f)
    }

    private fun detection(
        x: Float,
        y: Float,
        confidence: Float
    ) = DetectorTypes.DetectionResult(
        label = "wallet",
        confidence = confidence,
        normalizedX = x,
        normalizedY = y,
        normalizedArea = 0.10f,
        normalizedW = 0.25f,
        normalizedH = 0.40f,
        sourceW = 720,
        sourceH = 1280
    )
}
