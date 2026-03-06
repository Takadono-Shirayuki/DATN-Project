# ProGuard rules for GR2 Camera App

# Keep all classes in our package
-keep class com.gr2.camera.** { *; }

# Keep TensorFlow Lite
-keep class org.tensorflow.** { *; }

# Keep Gson
-keep class com.google.gson.** { *; }
-keep class * implements com.google.gson.JsonSerializer
-keep class * implements com.google.gson.JsonDeserializer

# Keep OkHttp
-keep class okhttp3.** { *; }
-keep class okio.** { *; }

# Keep Android classes
-keep class androidx.** { *; }

# Keep our models
-keepclassmembers class * {
    *** get*();
    void set*(***);
}
