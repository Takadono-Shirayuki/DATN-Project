package com.gr2.camera.network

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * WebSocket client for sending frames and metadata to server
 */
class NetworkManager(
    private val serverIP: String,
    private val serverPort: Int = 3000,
    private val onStatusChanged: (Boolean) -> Unit = {}
) : WebSocketListener() {

    companion object {
        private const val TAG = "NetworkManager"
    }

    private var webSocket: WebSocket? = null
    private var isConnected = false

    fun connect() {
        val url = "ws://$serverIP:$serverPort/camera"
        
        val client = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .pingInterval(10, TimeUnit.SECONDS)
            .build()

        val request = Request.Builder()
            .url(url)
            .build()

        try {
            webSocket = client.newWebSocket(request, this)
            Log.d(TAG, "Connecting to $url")
        } catch (e: Exception) {
            Log.e(TAG, "Connection failed: $e")
            onStatusChanged(false)
        }
    }

    override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
        super.onOpen(webSocket, response)
        isConnected = true
        Log.d(TAG, "WebSocket connected")
        onStatusChanged(true)

        // Send registration
        sendRegistration("gr2-camera-android", "recognition")
    }

    override fun onMessage(webSocket: WebSocket, text: String) {
        super.onMessage(webSocket, text)
        Log.d(TAG, "Received: $text")
    }

    override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
        super.onClosing(webSocket, code, reason)
        isConnected = false
        Log.d(TAG, "WebSocket closing: $code - $reason")
        webSocket.close(1000, null)  // complete the close handshake
        onStatusChanged(false)
    }

    override fun onFailure(webSocket: WebSocket, t: Throwable, response: okhttp3.Response?) {
        super.onFailure(webSocket, t, response)
        isConnected = false
        Log.e(TAG, "WebSocket error: $t")
        onStatusChanged(false)
    }

    /** Send a pre-encoded JPEG byte array — no second compression step. */
    fun sendFrameJpeg(jpegBytes: ByteArray) {
        if (!isConnected || webSocket == null) {
            Log.w(TAG, "Not connected")
            return
        }
        try {
            val base64 = Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
            val json = "{\"type\":\"frame\",\"frame\":\"$base64\"}"
            webSocket?.send(json)
        } catch (e: Exception) {
            Log.e(TAG, "Send frame failed: $e")
        }
    }

    @Deprecated("Use sendFrameJpeg to avoid double JPEG encode")
    fun sendFrame(
        bitmap: Bitmap,
        personCount: Int,
        inferenceMs: Long,
        metadata: Map<String, Any> = emptyMap()
    ) {
        val baos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 75, baos)
        sendFrameJpeg(baos.toByteArray())
    }

    fun sendRegistration(deviceName: String, deviceType: String) {
        if (!isConnected || webSocket == null) {
            Log.w(TAG, "Not connected")
            return
        }

        try {
            val message = mapOf(
                "type" to "register",
                "device_name" to deviceName,
                "device_type" to deviceType
            )
            val json = org.json.JSONObject(message).toString()
            webSocket?.send(json)
            Log.d(TAG, "Registration sent: $deviceName")
        } catch (e: Exception) {
            Log.e(TAG, "Registration failed: $e")
        }
    }

    fun disconnect() {
        webSocket?.close(1000, "Client disconnecting")
        isConnected = false
        Log.d(TAG, "Disconnected")
        onStatusChanged(false)
    }

    fun isConnected(): Boolean = isConnected
}
