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
import android.widget.Button
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
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

    private val objectDetector = ObjectDetector()

    @Volatile private var isSearching = false
    private var currentTarget = ""

    private var camera: Camera? = null
    @Volatile private var isTorchOn = false

    // Single-target tracker — replaces the old MISS_HOLD_FRAMES freeze with a
    // constant-velocity predictor + confidence decay. Bridges short occlusions
    // (~½ s) without holding a stale box during camera pans. See TargetTracker.kt.
    private val tracker = TargetTracker()

    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
        // Hysteresis thresholds (0–255 luma scale).
        // Wide gap is intentional: the torch itself raises luma, so a narrow gap causes flicker.
        private const val TORCH_ON_LUMA  = 80   // turn torch on  when avg luma drops below this
        private const val TORCH_OFF_LUMA = 170  // turn torch off when avg luma rises above this
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

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

        setupTts()
        setupSpeechRecognizer()
        setupObjectDetector()
        objectDetector.loadModel(this, "yolo26n.tflite")

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

    override fun onDestroy() {
        super.onDestroy()
        setTorch(false)
        tts.stop()
        tts.shutdown()
        speechRecognizer.destroy()
        audioFeedback.stop()
        objectDetector.close()
        cameraExecutor.shutdown()
    }

    // ── Setup helpers ─────────────────────────────────────────────────────────

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
                speak("Sorry, I did not catch that. Please try again.")
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
        objectDetector.setListener(object : DetectorContract.Listener {
            override fun onResult(result: DetectorContract.DetectionResult?) {
                if (!isSearching) return
                val displayed = if (result != null) {
                    tracker.onHit(result)
                    tracker.current()
                } else {
                    // On a miss the tracker advances its predicted position by the
                    // current velocity and decays confidence; returns null once it
                    // has fallen below CONF_FLOOR (occlusion has lasted too long).
                    tracker.onMiss()
                }

                audioFeedback.update(
                    displayed?.let {
                        AudioFeedbackManager.DetectionResult(
                            normalizedX    = it.normalizedX,
                            normalizedY    = it.normalizedY,
                            normalizedArea = it.normalizedArea,
                            confidence     = it.confidence
                        )
                    }
                )
                runOnUiThread {
                    boundingBoxOverlay.updateDetection(displayed)
                    statusTextView.text = if (displayed != null)
                        "${displayed.label} ${"%.0f".format(displayed.confidence * 100)}%"
                    else
                        "Searching for $currentTarget\u2026"
                }
            }
        })
    }

    // ── Camera ────────────────────────────────────────────────────────────────

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // Cap analyzer frames to ~720p. The camera otherwise hands us full sensor
            // resolution (~12 MP on the S23 Ultra), and toBitmap() pays a YUV→ARGB
            // memcpy proportional to that. The model letterboxes to 320×320 anyway,
            // so the extra pixels are pure overhead.
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
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        updateTorch(imageProxy)
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

    // ── User interaction ──────────────────────────────────────────────────────

    private fun onListenButtonTapped() {
        if (isSearching) {
            stopSearching()
            speak("Search stopped.")
            return
        }
        speak("Listening")
        Handler(Looper.getMainLooper()).postDelayed({
            speechRecognizer.startListening(speechIntent)
        }, 1000)
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
        objectDetector.setTarget(target)
        audioFeedback.start(target)
        speak("Searching for $target.")
        listenButton.text = getString(R.string.tap_to_stop)
        statusTextView.text = "Searching for $target\u2026"
    }

    private fun stopSearching() {
        isSearching = false
        tracker.reset()
        audioFeedback.stop()
        objectDetector.setTarget("")
        setTorch(false)
        listenButton.text = getString(R.string.tap_to_speak)
        statusTextView.text = getString(R.string.status_idle)
        boundingBoxOverlay.updateDetection(null)
    }

    // ── Permissions ───────────────────────────────────────────────────────────

    private fun onPermissionsReady() {
        startCamera()
        speak("Welcome to FindMe. Tap the screen to say what you want to find.")
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    // ── Torch / low-light ────────────────────────────────────────────────────

    /**
     * Called on the camera executor thread for every frame (before inference).
     * Samples a sparse grid of Y-plane pixels to compute average luma, then
     * enables or disables the torch with hysteresis to avoid rapid flickering.
     * Does nothing when not actively searching.
     */
    private fun updateTorch(image: ImageProxy) {
        if (!isSearching) return
        val plane  = image.planes.getOrNull(0) ?: return
        val buffer = plane.buffer
        val stride = plane.rowStride
        val w = image.width
        val h = image.height

        // Sample every 16th pixel — cheap but representative
        var sum = 0L; var count = 0
        var row = 0
        while (row < h) {
            var col = 0
            while (col < w) {
                val idx = row * stride + col
                if (idx < buffer.limit()) { sum += buffer.get(idx).toInt() and 0xFF; count++ }
                col += 16
            }
            row += 16
        }
        if (count == 0) return

        val luma = (sum / count).toInt()
        when {
            !isTorchOn && luma < TORCH_ON_LUMA  -> setTorch(true)
            isTorchOn  && luma > TORCH_OFF_LUMA -> setTorch(false)
        }
    }

    private fun setTorch(on: Boolean) {
        isTorchOn = on
        camera?.cameraControl?.enableTorch(on)
    }

    // ── TTS helper ────────────────────────────────────────────────────────────

    private fun speak(message: String) {
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
    }
}
