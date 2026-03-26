package com.example.findme

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlin.math.PI
import kotlin.math.sin

/**
 * Produces continuous directional beeping to guide a blind user toward a detected object.
 *
 * Encoding:
 *   - Pitch (frequency): maps horizontal position of the object in the camera frame.
 *       Left edge  → 220 Hz (low)
 *       Centre     → 440 Hz (medium)
 *       Right edge → 880 Hz (high)
 *
 *   - Tempo (beep interval): maps how large/close the object appears in the frame.
 *       Small / far  → 1 200 ms between beeps (slow)
 *       Large / near →   100 ms between beeps (rapid)
 *
 * When no detection result is available (object not yet found) the manager plays a slow,
 * neutral 440 Hz pulse so the user knows the search is active.
 */
class AudioFeedbackManager {

    data class DetectionResult(
        /** Horizontal centre of the bounding box: 0.0 = far left, 1.0 = far right. */
        val normalizedX: Float,
        /** Bounding-box area as a fraction of the total frame area: 0.0 = tiny, 1.0 = fills frame. */
        val normalizedArea: Float
    )

    @Volatile private var currentResult: DetectionResult? = null
    @Volatile private var isRunning = false
    private var beepThread: Thread? = null

    private val sampleRate = 44100

    fun start() {
        if (isRunning) return
        isRunning = true
        beepThread = Thread {
            while (isRunning) {
                val result = currentResult
                val frequencyHz = if (result != null) mapToFrequency(result.normalizedX) else 440f
                val intervalMs  = if (result != null) mapToInterval(result.normalizedArea) else 1200L

                playBeep(frequencyHz, durationMs = 80)

                try {
                    Thread.sleep(intervalMs)
                } catch (e: InterruptedException) {
                    break
                }
            }
        }.also { it.isDaemon = true; it.start() }
    }

    fun stop() {
        isRunning = false
        beepThread?.interrupt()
        beepThread = null
        currentResult = null
    }

    /** Call this from the detection callback to update direction/distance. Pass null when the
     *  target is not currently visible in the frame. */
    fun update(result: DetectionResult?) {
        currentResult = result
    }

    // ── Mapping helpers ──────────────────────────────────────────────────────

    /** 0.0 → 220 Hz,  0.5 → 550 Hz,  1.0 → 880 Hz (left-to-right). */
    private fun mapToFrequency(normalizedX: Float): Float =
        220f + normalizedX.coerceIn(0f, 1f) * 660f

    /** Large area (close) → short interval (fast),  small area (far) → long interval (slow). */
    private fun mapToInterval(normalizedArea: Float): Long =
        (1200L - (normalizedArea.coerceIn(0f, 1f) * 1100).toLong()).coerceAtLeast(100L)

    // ── Audio synthesis ───────────────────────────────────────────────────────

    private fun playBeep(frequency: Float, durationMs: Int) {
        val numSamples = sampleRate * durationMs / 1000
        val buffer = ShortArray(numSamples) { i ->
            val angle = 2.0 * PI * i * frequency / sampleRate
            (sin(angle) * Short.MAX_VALUE * 0.6).toInt().toShort()
        }

        val bufferSizeBytes = buffer.size * Short.SIZE_BYTES
        val audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufferSizeBytes)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        audioTrack.write(buffer, 0, buffer.size)
        audioTrack.play()
        try {
            Thread.sleep(durationMs.toLong())
        } catch (e: InterruptedException) {
            // Interrupted during playback — clean up and propagate
            audioTrack.stop()
            audioTrack.release()
            Thread.currentThread().interrupt()
            return
        }
        audioTrack.stop()
        audioTrack.release()
    }
}
