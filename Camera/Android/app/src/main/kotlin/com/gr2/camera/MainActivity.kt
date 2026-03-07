package com.gr2.camera

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Rect
import android.graphics.YuvImage
import android.graphics.ImageFormat
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.ProcessLifecycleOwner
import com.gr2.camera.camera.CameraManager
import com.gr2.camera.network.NetworkManager
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val PERMISSION_REQUEST_CODE = 100
    }

    private lateinit var cameraAreaLayout: FrameLayout
    private lateinit var previewView: PreviewView
    private lateinit var switchCameraButton: Button
    private lateinit var serverIPEditText: EditText
    private lateinit var serverPortEditText: EditText
    private lateinit var connectButton: Button
    private lateinit var statusTextView: TextView
    private lateinit var statsTextView: TextView

    private var cameraManager: CameraManager? = null
    private var networkManager: NetworkManager? = null

    private var isRunning = false
    private var frameCount = 0L
    private var lastStatsTime = System.currentTimeMillis()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initializeViews()
        checkPermissions()
    }

    private fun initializeViews() {
        cameraAreaLayout = findViewById(R.id.cameraAreaLayout)
        previewView = findViewById(R.id.previewView)
        switchCameraButton = findViewById(R.id.switchCameraButton)
        serverIPEditText = findViewById(R.id.serverIPEditText)
        serverPortEditText = findViewById(R.id.serverPortEditText)
        connectButton = findViewById(R.id.connectButton)
        statusTextView = findViewById(R.id.statusTextView)
        statsTextView = findViewById(R.id.statsTextView)

        connectButton.setOnClickListener { onConnectButtonClick() }
        switchCameraButton.setOnClickListener {
            cameraManager?.switchCamera()
        }
    }

    private fun checkPermissions() {
        val needed = arrayOf(Manifest.permission.CAMERA, Manifest.permission.INTERNET)
            .filter { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), PERMISSION_REQUEST_CODE)
        }
        // Camera starts only after connecting — no action on permission grant
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                updateStatus("Ready to connect")
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                updateStatus("Permission denied")
            }
        }
    }

    private fun startCamera() {
        // Bind camera to ProcessLifecycleOwner so it keeps running when app is in background
        cameraManager = CameraManager(this, ProcessLifecycleOwner.get(), previewView)
        cameraManager?.startCamera { image -> onFrameAvailable(image) }
    }

    private fun onFrameAvailable(image: ImageProxy) {
        if (!isRunning) return
        try {
            val jpeg = imageProxyToJpeg(image)
            networkManager?.sendFrameJpeg(jpeg)
            frameCount++
            updateStats()
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing failed: $e")
        }
    }

    /** Convert ImageProxy (YUV_420_888) → JPEG bytes in a single encode step. */
    private fun imageProxyToJpeg(image: ImageProxy): ByteArray {
        val planes = image.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val ySize = yPlane.buffer.remaining()
        // NV21 layout: Y plane (ySize bytes) + interleaved VU (ySize/2 bytes)
        val nv21 = ByteArray(ySize + image.width * image.height / 2)

        yPlane.buffer.get(nv21, 0, ySize)

        val uvPixelStride = uPlane.pixelStride
        if (uvPixelStride == 1) {
            // Planar U and V — interleave manually into NV21 (V first, then U)
            val u = ByteArray(uPlane.buffer.remaining())
            val v = ByteArray(vPlane.buffer.remaining())
            uPlane.buffer.get(u)
            vPlane.buffer.get(v)
            val uvSize = minOf(u.size, v.size)
            for (i in 0 until uvSize) {
                nv21[ySize + 2 * i]     = v[i]
                nv21[ySize + 2 * i + 1] = u[i]
            }
        } else {
            // pixelStride == 2: U and V share the same interleaved memory.
            // vPlane buffer starts at V0 → layout is V0,U0,V1,U1,... = NV21 order.
            val vBuffer = vPlane.buffer
            val uvData = ByteArray(vBuffer.remaining())
            vBuffer.get(uvData)
            System.arraycopy(uvData, 0, nv21, ySize, minOf(uvData.size, nv21.size - ySize))
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 75, out)
        return out.toByteArray()
    }

    private fun onConnectButtonClick() {
        val serverIP = serverIPEditText.text.toString().trim()
        val port = serverPortEditText.text.toString().toIntOrNull() ?: 3001

        if (serverIP.isEmpty()) {
            Toast.makeText(this, "Please enter server IP", Toast.LENGTH_SHORT).show()
            return
        }

        if (isRunning) stopConnection() else startConnection(serverIP, port)
    }

    private fun startConnection(serverIP: String, port: Int) {
        isRunning = true
        serverIPEditText.visibility = View.GONE
        serverPortEditText.visibility = View.GONE
        connectButton.text = "Stop"
        updateStatus("Connecting to $serverIP:$port…")

        networkManager = NetworkManager(serverIP, port) { connected ->
            runOnUiThread {
                if (connected) {
                    updateStatus("Connected → $serverIP:$port")
                    cameraAreaLayout.visibility = View.VISIBLE
                    startCamera()
                    ContextCompat.startForegroundService(
                        this, Intent(this, CameraForegroundService::class.java)
                    )
                } else {
                    // Unexpected disconnect while running
                    if (isRunning) stopConnection()
                }
            }
        }
        networkManager?.connect()
    }

    private fun stopConnection() {
        isRunning = false
        networkManager?.disconnect()
        cameraManager?.stopCamera()
        runOnUiThread {
            cameraAreaLayout.visibility = View.GONE
            statsTextView.text = ""
            connectButton.text = "Connect"
            serverIPEditText.visibility = View.VISIBLE
            serverPortEditText.visibility = View.VISIBLE
            updateStatus("Disconnected")
        }
        stopService(Intent(this, CameraForegroundService::class.java))
    }

    private fun updateStatus(message: String) {
        runOnUiThread { statusTextView.text = message }
    }

    private fun updateStats() {
        val now = System.currentTimeMillis()
        if (now - lastStatsTime > 1000) {
            val fps = frameCount * 1000 / (now - lastStatsTime)
            runOnUiThread { statsTextView.text = "FPS: $fps | Frames sent: $frameCount" }
            lastStatsTime = now
            frameCount = 0
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopConnection()
    }
}
