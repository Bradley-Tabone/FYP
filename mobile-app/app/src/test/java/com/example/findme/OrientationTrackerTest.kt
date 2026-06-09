package com.example.findme

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI

class OrientationTrackerTest {

    @Test
    fun currentYawIsNaNBeforeAnyEvent() {
        val tracker = OrientationTracker { }
        assertTrue(tracker.currentYawRadians().isNaN())
    }

    @Test
    fun feedingYawUpdatesStateAndCallsListener() {
        val seen = mutableListOf<Float>()
        val tracker = OrientationTracker { seen.add(it) }

        tracker.feedYawForTesting((PI / 2).toFloat())
        tracker.feedYawForTesting(0f)
        tracker.feedYawForTesting((-PI / 4).toFloat())

        assertEquals(3, seen.size)
        assertEquals((PI / 2).toFloat(), seen[0], 1e-6f)
        assertEquals(0f, seen[1], 1e-6f)
        assertEquals((-PI / 4).toFloat(), seen[2], 1e-6f)
        assertEquals((-PI / 4).toFloat(), tracker.currentYawRadians(), 1e-6f)
    }

    @Test
    fun nanYawCanBeFedExplicitly() {
        val seen = mutableListOf<Float>()
        val tracker = OrientationTracker { seen.add(it) }

        tracker.feedYawForTesting(Float.NaN)

        assertEquals(1, seen.size)
        assertTrue(seen[0].isNaN())
        assertTrue(tracker.currentYawRadians().isNaN())
    }
}
