package com.example.findme

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

/*
 * Listens to the phone's movement sensor and classifies how fast the phone is moving.
 * The app uses this to know whether the user is stationary, moving normally,
 * or moving too fast for the detector to reliably find the target.
 */
class MotionDetector : SensorEventListener {

    enum class MotionState { STATIONARY, MOVING, MOVING_TOO_FAST }

    private val sensorManager: SensorManager?
    private val sensor: Sensor?
    private val usesRawAccelerometer: Boolean
    private val onStateChanged: (MotionState) -> Unit

    constructor(context: Context, onStateChanged: (MotionState) -> Unit) {
        this.onStateChanged = onStateChanged
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        val linear = sensorManager?.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        if (linear != null) {
            sensor = linear
            usesRawAccelerometer = false
        } else {
            sensor = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            usesRawAccelerometer = sensor != null
        }
    }

    // Test constructor that lets unit tests feed fake movement values without a real sensor.
    internal constructor(onStateChanged: (MotionState) -> Unit) {
        this.onStateChanged = onStateChanged
        this.sensorManager = null
        this.sensor = null
        this.usesRawAccelerometer = false
    }

    @Volatile private var running = false
    @Volatile private var state: MotionState = MotionState.STATIONARY

    // Smoothed movement strength. NaN means no sensor value has been read yet.
    private var smoothedMag = Float.NaN

    /*
     * Estimated gravity on each phone axis.
     * Only used when the device does not provide linear acceleration directly.
     */
    private var gravityX = 0f
    private var gravityY = 0f
    private var gravityZ = 0f
    private var gravityInitialised = false

    /*
     * Timers used to avoid changing state because of one tiny spike or pause.
     * A condition must stay true for long enough before the state changes.
     */
    private var fastSinceMs = 0L
    private var stillSinceMs = 0L
    private var cooldownSinceMs = 0L

    companion object {
        // Smoothing strength for movement readings.
        private const val MOTION_ALPHA = 0.20f

        // Smoothing strength for gravity estimation when using the raw accelerometer.
        private const val GRAVITY_ALPHA = 0.10f

        // Movement below this is treated as basically still.
        private const val STILL_THRESHOLD = 0.25f

        // Movement above this may be too fast for reliable detection.
        private const val FAST_THRESHOLD = 3.5f

        // The phone must slow below this before leaving MOVING_TOO_FAST.
        private const val FAST_EXIT_THRESHOLD = FAST_THRESHOLD * 0.7f

        // How long calm movement must last before becoming STATIONARY.
        private const val STILL_REQUIRED_MS = 2_500L

        // How long fast movement must last before becoming MOVING_TOO_FAST.
        private const val FAST_REQUIRED_MS = 600L

        // How long movement must stay slower before leaving MOVING_TOO_FAST.
        private const val COOLDOWN_MS = 1_200L
    }

    // Start listening for phone movement updates.
    fun start() {
        if (running) return
        val sm = sensorManager ?: return
        val s = sensor ?: return
        running = sm.registerListener(this, s, SensorManager.SENSOR_DELAY_UI)
    }

    // Stop listening and reset the movement state.
    fun stop() {
        if (!running) return
        sensorManager?.unregisterListener(this)
        running = false
        resetInternalState()
    }

    override fun onSensorChanged(event: SensorEvent) {
        // Android gives X, Y, and Z movement. Convert them into one magnitude.
        val mag = magnitudeOf(event.values)
        feed(mag, System.currentTimeMillis())
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // Movement calculation and classification logic.

    private fun magnitudeOf(values: FloatArray): Float {
        val x: Float; val y: Float; val z: Float
        if (usesRawAccelerometer) {
            if (!gravityInitialised) {
                gravityX = values[0]; gravityY = values[1]; gravityZ = values[2]
                gravityInitialised = true
            } else {
                gravityX = gravityX * (1f - GRAVITY_ALPHA) + values[0] * GRAVITY_ALPHA
                gravityY = gravityY * (1f - GRAVITY_ALPHA) + values[1] * GRAVITY_ALPHA
                gravityZ = gravityZ * (1f - GRAVITY_ALPHA) + values[2] * GRAVITY_ALPHA
            }
            // Remove estimated gravity so the remaining values mostly represent movement.
            x = values[0] - gravityX
            y = values[1] - gravityY
            z = values[2] - gravityZ
        } else {
            x = values[0]; y = values[1]; z = values[2]
        }
        // Convert X, Y, and Z movement into one overall movement-strength number.
        return sqrt(x * x + y * y + z * z)
    }

    private fun feed(rawMag: Float, now: Long) {
        smoothedMag = if (smoothedMag.isNaN()) {
            rawMag
        } else {
            smoothedMag * (1f - MOTION_ALPHA) + rawMag * MOTION_ALPHA
        }

        // Work out the next movement state and report it only if it changed.
        val next = classify(smoothedMag, now)
        if (next != state) {
            state = next
            onStateChanged(next)
        }
    }

    /*
     * Turns movement strength into STATIONARY, MOVING, or MOVING_TOO_FAST.
     * The timer checks stop the app reacting to one-off spikes or tiny pauses.
     */
    private fun classify(mag: Float, now: Long): MotionState {
        return when (state) {
            MotionState.STATIONARY -> {
                if (mag > FAST_THRESHOLD) {
                    if (fastSinceMs == 0L) fastSinceMs = now
                    if (now - fastSinceMs >= FAST_REQUIRED_MS) {
                        stillSinceMs = 0L; cooldownSinceMs = 0L
                        MotionState.MOVING_TOO_FAST
                    } else {
                        MotionState.STATIONARY
                    }
                } else {
                    fastSinceMs = 0L
                    if (mag > STILL_THRESHOLD) {
                        stillSinceMs = 0L
                        MotionState.MOVING
                    } else {
                        MotionState.STATIONARY
                    }
                }
            }
            MotionState.MOVING -> {
                if (mag > FAST_THRESHOLD) {
                    if (fastSinceMs == 0L) fastSinceMs = now
                    if (now - fastSinceMs >= FAST_REQUIRED_MS) {
                        stillSinceMs = 0L; cooldownSinceMs = 0L
                        MotionState.MOVING_TOO_FAST
                    } else {
                        MotionState.MOVING
                    }
                } else {
                    fastSinceMs = 0L
                    if (mag < STILL_THRESHOLD) {
                        if (stillSinceMs == 0L) stillSinceMs = now
                        if (now - stillSinceMs >= STILL_REQUIRED_MS) {
                            MotionState.STATIONARY
                        } else {
                            MotionState.MOVING
                        }
                    } else {
                        stillSinceMs = 0L
                        MotionState.MOVING
                    }
                }
            }
            MotionState.MOVING_TOO_FAST -> {
                if (mag < FAST_EXIT_THRESHOLD) {
                    if (cooldownSinceMs == 0L) cooldownSinceMs = now
                    if (now - cooldownSinceMs >= COOLDOWN_MS) {
                        fastSinceMs = 0L
                        if (mag < STILL_THRESHOLD) MotionState.STATIONARY else MotionState.MOVING
                    } else {
                        MotionState.MOVING_TOO_FAST
                    }
                } else {
                    cooldownSinceMs = 0L
                    MotionState.MOVING_TOO_FAST
                }
            }
        }
    }

    private fun resetInternalState() {
        smoothedMag = Float.NaN
        gravityInitialised = false
        fastSinceMs = 0L
        stillSinceMs = 0L
        cooldownSinceMs = 0L
        state = MotionState.STATIONARY
    }

    // Testing helpers.

    internal fun feedForTesting(magnitude: Float, nowMs: Long) {
        feed(magnitude, nowMs)
    }

    internal fun stateForTesting(): MotionState = state

    internal fun resetForTesting() {
        resetInternalState()
    }
}
