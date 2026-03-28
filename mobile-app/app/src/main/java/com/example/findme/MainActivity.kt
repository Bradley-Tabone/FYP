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
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
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

    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
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
        tts.stop()
        tts.shutdown()
        speechRecognizer.destroy()
        audioFeedback.stop()
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
        objectDetector.setListener(object : ObjectDetector.Listener {
            override fun onResult(result: ObjectDetector.DetectionResult?) {
                if (!isSearching) return
                audioFeedback.update(
                    result?.let {
                        AudioFeedbackManager.DetectionResult(it.normalizedX, it.normalizedArea)
                    }
                )
                runOnUiThread {
                    boundingBoxOverlay.updateDetection(result)
                    statusTextView.text = if (result != null) {
                        "${result.label} • ${"%.0f".format(result.confidence * 100)}%"
                    } else {
                        "Searching for $currentTarget\u2026"
                    }
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

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        objectDetector.analyze(imageProxy)
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
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
            command.contains("wallet")                              -> startSearching("wallet")
            command.contains("key")                                 -> startSearching("keys")
            command.contains("glasses") || command.contains("spectacles") -> startSearching("glasses")
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
        audioFeedback.stop()
        objectDetector.setTarget("")
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

    // ── TTS helper ────────────────────────────────────────────────────────────

    private fun speak(message: String) {
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
    }
}
