# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# ───────────────────────────────────────────────────────────────────────────
# Keep rules for the on-device ML stack. R8 would otherwise strip native-
# bridge classes whose only references are JNI / reflection from the .so
# files, which crashes the app at first inference call ("class not found").
# ───────────────────────────────────────────────────────────────────────────

# TensorFlow Lite — the interpreter, GPU/NNAPI delegates, and op kernels
# are looked up reflectively from native code.
-keep class org.tensorflow.lite.** { *; }
-keep interface org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# CameraX — uses reflection internally for some lifecycle / config classes.
-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**