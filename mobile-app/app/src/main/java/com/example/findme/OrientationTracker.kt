package com.example.findme

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

/**
 Listens to the phone's rotation sensor and keeps track of yaw.
 
 Yaw is the left/right direction the phone is facing. The app uses this
 later to remember where the target was last seen and give clock-style
 guidance when the target leaves the camera view.
 */
class OrientationTracker : SensorEventListener {

    private val sensorManager: SensorManager?
    private val sensor: Sensor?
    private val onYawChanged: (Float) -> Unit

    /**  Temporary matrices used when converting raw sensor values into yaw.
    They are created once and reused so every sensor update does not need
    to create fresh arrays.*/
    private val rotationMatrix = FloatArray(9)
    private val remappedMatrix = FloatArray(9)
    private val orientation = FloatArray(3)

    @Volatile private var running = false
    @Volatile private var yawRad: Float = Float.NaN

    constructor(context: Context, onYawChanged: (Float) -> Unit) {
        this.onYawChanged = onYawChanged
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        sensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
    }

    /** Test constructor that lets unit tests feed fake yaw values without a real phone sensor. */
    internal constructor(onYawChanged: (Float) -> Unit) {
        this.onYawChanged = onYawChanged
        this.sensorManager = null
        this.sensor = null
    }

    /** Start listening for phone rotation updates. */
    fun start() {
        if (running) return
        val sm = sensorManager ?: return
        val s = sensor ?: return
        running = sm.registerListener(this, s, SensorManager.SENSOR_DELAY_UI)
    }

    /** Stop listening and clear the current yaw value. */
    fun stop() {
        if (!running) return
        sensorManager?.unregisterListener(this)
        running = false
        yawRad = Float.NaN
    }

    /** Return the latest yaw value, or NaN if we do not have one yet. */
    fun currentYawRadians(): Float = yawRad

    override fun onSensorChanged(event: SensorEvent) {
        SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
        /*  Adjust Android's raw sensor axes so the yaw value matches the phone
        being held upright with the back camera pointing forward.*/
        SensorManager.remapCoordinateSystem(
            rotationMatrix,
            SensorManager.AXIS_X,
            SensorManager.AXIS_Z,
            remappedMatrix
        )
        SensorManager.getOrientation(remappedMatrix, orientation)
        val newYaw = orientation[0]
        yawRad = newYaw
        onYawChanged(newYaw)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // Testing helper.

    internal fun feedYawForTesting(yawRadians: Float) {
        yawRad = yawRadians
        onYawChanged(yawRadians)
    }
}
