plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.findme"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.example.findme"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Strip x86/armeabi-v7a native libs — this app targets real Android phones only.
        // Reduces APK size by ~60% of native-lib weight (arm64-v8a is all modern devices).
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        release {
            // R8 minification + resource shrinking. Strips dead code (unused
            // TFLite ops, unused Kotlin stdlib classes, unused R-class entries)
            // and obfuscates names — both shrink the APK noticeably. The
            // -keep rules in proguard-rules.pro pin TFLite, ORT, and the
            // CameraX/JNI entry points that are reflected at runtime.
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    packaging {
        jniLibs {
            // Compress native libs so 16 KB page-size alignment is not required.
            // Kept as a safety net for third-party .so files that may not yet be aligned.
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)

    // CameraX
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)

    // LiteRT — Google's successor to TFLite (same org.tensorflow.lite.* package names,
    // API-compatible drop-in). Version 1.4.0 ships updated GPU kernels with broader op
    // coverage and fixes the GpuDelegateFactory$Options packaging bug present in 1.0.x.
    implementation(libs.litert)
    implementation(libs.litert.gpu)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}