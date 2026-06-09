package com.example.findme

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class DetectionOutputParserTest {

    @Test
    fun parsesEndToEndNormalizedOutput() {
        val output = arrayOf(
            arrayOf(
                floatArrayOf(0.25f, 0.25f, 0.75f, 0.75f, 0.90f, 2f),
                floatArrayOf(0.10f, 0.10f, 0.20f, 0.20f, 0.95f, 0f)
            )
        )

        val result = DetectionOutputParser.parse("wallet", output, 0, 0, 640, 640)

        assertNotNull(result)
        assertEquals("wallet", result!!.label)
        assertEquals(0.90f, result.confidence, 0.001f)
        assertEquals(0.50f, result.normalizedX, 0.001f)
        assertEquals(0.50f, result.normalizedY, 0.001f)
        assertEquals(0.50f, result.normalizedW, 0.001f)
        assertEquals(0.50f, result.normalizedH, 0.001f)
    }

    @Test
    fun parsesEndToEndAbsolutePixelOutput() {
        val output = arrayOf(
            arrayOf(floatArrayOf(160f, 160f, 480f, 480f, 0.90f, 2f))
        )

        val result = DetectionOutputParser.parse("wallet", output, 0, 0, 640, 640)

        assertNotNull(result)
        assertEquals(0.50f, result!!.normalizedX, 0.001f)
        assertEquals(0.50f, result.normalizedY, 0.001f)
        assertEquals(0.50f, result.normalizedW, 0.001f)
        assertEquals(0.50f, result.normalizedH, 0.001f)
    }

    @Test
    fun returnsNullForUnknownTarget() {
        val output = arrayOf(
            arrayOf(floatArrayOf(0.25f, 0.25f, 0.75f, 0.75f, 0.90f, 2f))
        )

        assertNull(DetectionOutputParser.parse("phone", output, 0, 0, 640, 640))
    }
}
