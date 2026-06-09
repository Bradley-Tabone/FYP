package com.example.findme

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var tts: TextToSpeech
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechIntent: Intent
    private lateinit var previewView: PreviewView
    private lateinit var listenButton: Button
    private lateinit var statusTextView: TextView
    private lateinit var boundingBoxOverlay: BoundingBoxOverlay
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var audioFeedback: AudioFeedbackManager
    private lateinit var motionDetector: MotionDetector
    private lateinit var orientationTracker: OrientationTracker

    private val objectDetector = ObjectDetector()

    private val uiHandler = Handler(Looper.getMainLooper())
    private val startListeningRunnable = Runnable { speechRecognizer.startListening(speechIntent) }

    @Volatile private var isSearching = false
    private var currentTarget = ""

    private var camera: Camera? = null
    @Volatile private var isTorchOn = false
    @Volatile private var smoothedTorchLuma = Float.NaN
    @Volatile private var torchDarkSinceMs = 0L
    @Volatile private var torchBrightSinceMs = 0L
    @Volatile private var torchTurnedOnAtMs = 0L

    // Smooths the visible box during very short missed detections.
    private val tracker = TargetTracker()

    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
        /*
         * Torch brightness values use luma: 0 is dark, 255 is bright.
         * The timing constants stop the torch flickering on borderline frames.
         */
        private const val TORCH_ON_LUMA = 75
        private const val TORCH_OFF_LUMA = 180
        private const val TORCH_LUMA_ALPHA = 0.20f
        private const val TORCH_ON_REQUIRED_MS = 600L
        private const val TORCH_OFF_REQUIRED_MS = 3_000L
        private const val MIN_TORCH_ON_MS = 8_000L
    }

    // Lifecycle

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        previewView        = findViewById(R.id.previewView)
        listenButton       = findViewById(R.id.listenButton)
        statusTextView     = findViewById(R.id.statusTextView)
        boundingBoxOverlay = findViewById(R.id.boundingBoxOverlay)
        cameraExecutor     = Executors.newSingleThreadExecutor()
        audioFeedback      = AudioFeedbackManager { msg -> speak(msg) }
        motionDetector     = MotionDetector(applicationContext) { state ->
            audioFeedback.updateMotion(state)
        }
        orientationTracker = OrientationTracker(applicationContext) { yaw ->
            audioFeedback.updateOrientation(yaw)
        }

        setupTts()
        setupSpeechRecognizer()
        setupObjectDetector()

        /*
         * Load the model in the camera executor and keep the listen button disabled
         * until the model is ready.
         */
        listenButton.isEnabled = false
        cameraExecutor.execute {
            objectDetector.loadModel(applicationContext, "yolo26n.tflite")
            // Only re-enable the button if the activity still exists.
            runOnUiThread { if (!isDestroyed) listenButton.isEnabled = true }
        }

        listenButton.setOnClickListener { onListenButtonTapped() }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        if (allPermissionsGranted()) {
            onPermissionsReady()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, PERMISSION_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (allPermissionsGranted()) {
                onPermissionsReady()
            } else {
                speak("Required permissions were denied. Please enable camera and microphone in settings.")
            }
        }
    }

    override fun onPause() {
        uiHandler.removeCallbacks(startListeningRunnable)
        if (isSearching) stopSearching()
        speechRecognizer.cancel()
        setTorch(false)
        setKeepScreenAwake(false)
        motionDetector.stop()
        orientationTracker.stop()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        uiHandler.removeCallbacks(startListeningRunnable)
        setTorch(false)
        setKeepScreenAwake(false)
        tts.stop()
        tts.shutdown()
        speechRecognizer.destroy()
        audioFeedback.stop()
        motionDetector.stop()
        orientationTracker.stop()
        /*
         * Close the detector on the same executor so cleanup waits behind any
         * model-loading work already queued.
         */
        cameraExecutor.execute { objectDetector.close() }
        cameraExecutor.shutdown()
    }

    // Setup helpers

    private fun setupTts() {
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale.UK
            }
        }
    }

    private fun setupSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.UK)
        }
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: Bundle?) {
                val spoken = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.get(0)
                    ?.lowercase()
            if (spoken != null) handleCommand(spoken)
        }
        override fun onError(error: Int) {
                when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH,
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> speak("I did not catch that. Please try again.")
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> speak("Microphone permission is needed to listen.")
                    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> {
                        speechRecognizer.cancel()
                    }
                    else -> speak("Listening did not work. Please try again.")
                }
            }
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    private fun setupObjectDetector() {
        objectDetector.setListener(object : DetectorTypes.Listener {
            override fun onLuma(luma: Int) {
                // Use frame brightness from ObjectDetector to control the torch while searching.
                if (!isSearching) return
                updateTorchFromLuma(luma, System.currentTimeMillis())
            }

            override fun onResult(result: DetectorTypes.DetectionResult?) {
                if (!isSearching) return
                val audioResult = result?.let {
                    AudioFeedbackManager.DetectionResult(
                        normalizedX    = it.normalizedX,
                        normalizedY    = it.normalizedY,
                        normalizedArea = it.normalizedArea,
                        confidence     = it.confidence
                    )
                }
                val displayed = if (result != null) {
                    tracker.onHit(result)
                    tracker.current()
                } else {
                    // If the detector misses, let the tracker predict briefly before clearing the box.
                    tracker.onMiss()
                }

                /*
                 * Audio uses only real detections.
                 * The overlay may show tracker predictions, but speech should not follow stale boxes.
                 */
                audioFeedback.update(audioResult)
                runOnUiThread {
                    boundingBoxOverlay.updateDetection(displayed)
                    statusTextView.text = if (displayed != null)
                        "${displayed.label} ${(displayed.confidence * 100).toInt()}%"
                    else
                        "Searching for $currentTarget\u2026"
                }
            }
        })
    }

    // Camera

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            /*
             * Keep analysis near 720p so frame decoding stays cheaper.
             * The model input is only 640x640, so much larger frames add overhead.
             */
            val analyzerResolution = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(720, 1280),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                    )
                )
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(analyzerResolution)
                .build()
                .also {
                    // ObjectDetector also reports brightness, so no extra luma scan is needed here.
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        objectDetector.analyze(imageProxy)
                    }
                }

            cameraProvider.unbindAll()
            camera = cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    // User interaction

    private fun onListenButtonTapped() {
        if (isSearching) {
            stopSearching()
            speak("Search stopped.")
            return
        }
        speak("Listening")
        uiHandler.postDelayed(startListeningRunnable, 1000)
    }

    private fun handleCommand(command: String) {
        when {
            command.contains("wallet")                                        -> startSearching("wallet")
            command.contains("key")                                           -> startSearching("keys")
            command.contains("glasses") || command.contains("spectacles")     -> startSearching("sunglasses")
            else -> speak("Object not recognised. Please say wallet, keys, or glasses.")
        }
    }

    private fun startSearching(target: String) {
        currentTarget = target
        isSearching = true
        setKeepScreenAwake(true)
        objectDetector.setTarget(target)
        audioFeedback.start(target)
        motionDetector.start()
        orientationTracker.start()
        speak("Searching for $target.")
        listenButton.text = getString(R.string.tap_to_stop)
        statusTextView.text = "Searching for $target\u2026"
    }

    private fun stopSearching() {
        isSearching = false
        tracker.reset()
        audioFeedback.stop()
        motionDetector.stop()
        orientationTracker.stop()
        objectDetector.setTarget("")
        setTorch(false)
        setKeepScreenAwake(false)
        listenButton.text = getString(R.string.tap_to_speak)
        statusTextView.text = getString(R.string.status_idle)
        boundingBoxOverlay.updateDetection(null)
    }

    private fun setKeepScreenAwake(awake: Boolean) {
        val updateFlag = {
            if (awake) {
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            } else {
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }

        if (Looper.myLooper() == Looper.getMainLooper()) {
            updateFlag()
        } else {
            runOnUiThread(updateFlag)
        }
    }

    // Permissions

    private fun onPermissionsReady() {
        startCamera()
        speak("Welcome to FindMe. Tap the screen to say what you want to find.")
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    // Torch / low-light

    /*
     * Uses smoothed frame brightness to turn the torch on or off.
     * Timing checks stop the torch flickering in borderline lighting.
     */
    private fun updateTorchFromLuma(luma: Int, now: Long) {
        smoothedTorchLuma = if (smoothedTorchLuma.isNaN()) {
            luma.toFloat()
        } else {
            smoothedTorchLuma * (1f - TORCH_LUMA_ALPHA) + luma * TORCH_LUMA_ALPHA
        }

        if (!isTorchOn) {
            torchBrightSinceMs = 0L
            if (smoothedTorchLuma < TORCH_ON_LUMA) {
                if (torchDarkSinceMs == 0L) torchDarkSinceMs = now
                if (now - torchDarkSinceMs >= TORCH_ON_REQUIRED_MS) {
                    setTorch(true, now)
                }
            } else {
                torchDarkSinceMs = 0L
            }
            return
        }

        torchDarkSinceMs = 0L
        if (now - torchTurnedOnAtMs < MIN_TORCH_ON_MS) {
            torchBrightSinceMs = 0L
            return
        }

        if (smoothedTorchLuma > TORCH_OFF_LUMA) {
            if (torchBrightSinceMs == 0L) torchBrightSinceMs = now
            if (now - torchBrightSinceMs >= TORCH_OFF_REQUIRED_MS) {
                setTorch(false, now)
            }
        } else {
            torchBrightSinceMs = 0L
        }
    }

    private fun setTorch(on: Boolean, now: Long = System.currentTimeMillis()) {
        isTorchOn = on
        if (on) {
            torchTurnedOnAtMs = now
            torchDarkSinceMs = 0L
            torchBrightSinceMs = 0L
        } else {
            resetTorchTiming()
        }
        camera?.cameraControl?.enableTorch(on)
    }

    private fun resetTorchTiming() {
        smoothedTorchLuma = Float.NaN
        torchDarkSinceMs = 0L
        torchBrightSinceMs = 0L
        torchTurnedOnAtMs = 0L
    }

    // TTS helper

    private fun speak(message: String) {
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
    }
}
