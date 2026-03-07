package com.gr2.camera.camera

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors

class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView
) {
    companion object {
        private const val TAG = "CameraManager"
    }

    private var imageAnalysisCallback: ((image: ImageProxy) -> Unit)? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var preview: Preview? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK

    fun startCamera(
        onFrameAvailable: (image: ImageProxy) -> Unit
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val provider: ProcessCameraProvider = cameraProviderFuture.get()
            cameraProvider = provider
            this.imageAnalysisCallback = onFrameAvailable
            bindCamera(provider)
        }, { executor ->
            Executors.newSingleThreadExecutor().execute(executor)
        })
    }

    fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
            CameraSelector.LENS_FACING_FRONT
        else
            CameraSelector.LENS_FACING_BACK
        cameraProvider?.let { bindCamera(it) }
    }

    private fun bindCamera(provider: ProcessCameraProvider) {
        Handler(Looper.getMainLooper()).post {
            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()

            // Check the requested lens is actually available
            if (!provider.hasCamera(cameraSelector)) {
                Log.w(TAG, "Requested lens not available, falling back to back camera")
                lensFacing = CameraSelector.LENS_FACING_BACK
            }

            val selector = CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            this.preview = preview

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also {
                    it.setAnalyzer(
                        Executors.newSingleThreadExecutor(),
                        { image ->
                            imageAnalysisCallback?.invoke(image)
                            image.close()
                        }
                    )
                }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    selector,
                    preview,
                    imageAnalysis
                )
                Log.d(TAG, "Camera bound (lensFacing=$lensFacing)")
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }
    }

    fun stopCamera() {
        imageAnalysisCallback = null
        preview = null
        Handler(Looper.getMainLooper()).post {
            cameraProvider?.unbindAll()
            cameraProvider = null
            Log.d(TAG, "Camera stopped")
        }
    }

    /** Call from Activity.onPause() so CameraX doesn't fight a destroyed window surface. */
    fun detachPreviewSurface() {
        preview?.setSurfaceProvider(null)
    }

    /** Call from Activity.onResume() to restore live preview. */
    fun reattachPreviewSurface() {
        preview?.setSurfaceProvider(previewView.surfaceProvider)
    }
}
