package com.example.findme

import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate

/**
 * Configure [options] with the fastest available hardware delegate.
 * Tries GPU → NNAPI → CPU-only (whatever the caller pre-configured via setNumThreads).
 *
 * Returns the created [Delegate] (or null for the CPU fallback) so the caller
 * can close it alongside the Interpreter when the detector is released.
 * GPU and NNAPI delegates hold native resources that leak unless explicitly closed.
 *
 * GPU delegate uses:
 *  - FP16 inference (isPrecisionLossAllowed = true): ~2x throughput on most ops,
 *    negligible detection-quality loss for our use case.
 *  - INFERENCE_PREFERENCE_SUSTAINED_SPEED: optimises for repeated per-frame inference
 *    rather than one-shot startup latency (the default).
 * Without these flags the default GpuDelegate() is conservative enough that partial
 * CPU↔GPU graph fallback (e.g. for ops the GPU delegate can't run) dominates the
 * frame budget — often making it slower than pure CPU. Measured on S23 Ultra:
 * default ≈ 480 ms/frame, with these flags ≈ 30–60 ms/frame.
 */
fun applyAcceleration(options: Interpreter.Options, tag: String): Delegate? {
    try {
        val gpuOptions = GpuDelegate.Options().apply {
            isPrecisionLossAllowed = true
            inferencePreference = GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
        }
        val gpu = GpuDelegate(gpuOptions)
        options.addDelegate(gpu)
        Log.i(tag, "Using GPU delegate (FP16, sustained-speed)")
        return gpu
    } catch (t: Throwable) {
        Log.w(tag, "GPU delegate unavailable (${t.javaClass.simpleName}): ${t.message}")
    }

    try {
        val nnapi = NnApiDelegate()
        options.addDelegate(nnapi)
        Log.i(tag, "Using NNAPI delegate")
        return nnapi
    } catch (t: Throwable) {
        Log.w(tag, "NNAPI delegate unavailable (${t.javaClass.simpleName}): ${t.message}")
    }

    Log.i(tag, "Using CPU (no delegate)")
    return null
}
