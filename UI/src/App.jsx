import { useState, useEffect, useRef } from 'react'
import './App.css'
import CameraList from './components/CameraList'

function App() {
  const [hasFrame, setHasFrame] = useState(false)
  const [wsStatus, setWsStatus] = useState('connecting')
  const [cameras, setCameras] = useState([])
  const [selectedCamera, setSelectedCamera] = useState(null)
  const [localIP, setLocalIP] = useState('...')
  const [videoPath, setVideoPath] = useState('')
  const [videoEndedMap, setVideoEndedMap] = useState({}) // device_label -> { device, path }
  const videoPathsRef = useRef({}) // device_label -> full path, for replay
  const imgRef = useRef(null)
  const wsRef = useRef(null)
  const seenIPs = useRef(new Set())
  // Mirror selectedCamera in a ref so the WS closure always reads the current value.
  const selectedCameraRef = useRef(null)

  // Keep ref in sync with state so the WS closure sees the latest value.
  useEffect(() => { selectedCameraRef.current = selectedCamera }, [selectedCamera])

  // Fetch local machine IP from vite plugin
  useEffect(() => {
    fetch('/local-ip')
      .then(r => r.json())
      .then(d => { if (d.ips?.length) setLocalIP(d.ips[0]) })
      .catch(() => {})
  }, [])

  // Connect to RecognitionServer /monitor
  useEffect(() => {
    let mounted = true

    const connect = () => {
      if (!mounted) return
      setWsStatus('connecting')
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.hostname}:3001/monitor`)
      wsRef.current = ws

      ws.onopen = () => {
        if (mounted) setWsStatus('connected')
      }

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === 'frame' && msg.frame) {
            const device = msg.metadata?.device

            // Track cameras regardless of which is selected.
            if (device && !seenIPs.current.has(device)) {
              seenIPs.current.add(device)
              setCameras(prev => [...prev, device])
              setSelectedCamera(prev => prev ?? device)
            }

            // Only display the frame for the currently selected camera.
            const activeCam = selectedCameraRef.current
            if (imgRef.current && (!activeCam || device === activeCam)) {
              imgRef.current.src = `data:image/jpeg;base64,${msg.frame}`
              if (!hasFrame) setHasFrame(true)
            }
          } else if (msg.type === 'error') {
            // error displayed via videoEndedInfo or ignored
          } else if (msg.type === 'video_ended') {
            const dev = msg.device
            const fullPath = videoPathsRef.current[dev] ?? null
            setVideoEndedMap(prev => ({ ...prev, [dev]: { device: dev, path: fullPath } }))
          }
        } catch { /* ignore malformed */ }
      }

      ws.onerror = () => { if (mounted) setWsStatus('disconnected') }
      ws.onclose = () => {
        if (mounted) {
          setWsStatus('disconnected')
          setTimeout(connect, 3000)
        }
      }
    }

    connect()
    return () => {
      mounted = false
      wsRef.current?.close()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const sendVideoPath = () => {
    const path = videoPath.trim()
    if (!path) return
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      return
    }
    // Derive device_label the same way the server does: video:<basename>
    const basename = path.replace(/\\/g, '/').split('/').pop()
    const deviceLabel = `video:${basename}`
    videoPathsRef.current[deviceLabel] = path
    setVideoEndedMap(prev => { const n = { ...prev }; delete n[deviceLabel]; return n })
    wsRef.current.send(JSON.stringify({ type: 'video_path', path }))
  }

  const replayVideo = (path) => {
    if (!path) return
    if (wsRef.current?.readyState !== WebSocket.OPEN) return
    const basename = path.replace(/\\/g, '/').split('/').pop()
    const deviceLabel = `video:${basename}`
    videoPathsRef.current[deviceLabel] = path
    setVideoEndedMap(prev => { const n = { ...prev }; delete n[deviceLabel]; return n })
    wsRef.current.send(JSON.stringify({ type: 'video_path', path }))
  }

  const openFilePicker = () => {
    fetch('/open-file-dialog')
      .then(r => r.json())
      .then(d => { if (d.path) setVideoPath(d.path) })
      .catch(() => {})
  }

  const removeCamera = (ip) => {
    seenIPs.current.delete(ip)
    setCameras(prev => prev.filter(c => c !== ip))
    if (selectedCamera === ip) setSelectedCamera(null)
    setVideoEndedMap(prev => { const n = { ...prev }; delete n[ip]; return n })
  }

  const wsLabel = {
    connecting:   '⟳ Đang kết nối',
    connected:    '● Đã kết nối',
    disconnected: '○ Mất kết nối',
  }[wsStatus]

  return (
    <div className="app">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <span className="logo">Gait Recognition</span>
          <span className="subtitle">Hệ thống nhận dạng dáng đi từ camera</span>
          <span className={`ws-badge ws-${wsStatus}`}>{wsLabel}</span>
        </div>

        <CameraList
          cameras={cameras}
          selected={selectedCamera}
          onSelect={setSelectedCamera}
          onRemove={removeCamera}
        />

        {/* ── Video file debug ── */}
        <div className="video-debug">
          <span className="video-debug-label">Phát video từ máy chủ</span>
          <div className="video-input-row">
            <input
              className="video-path-input"
              type="text"
              placeholder="C:\videos\test.mp4"
              value={videoPath}
              onChange={e => setVideoPath(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendVideoPath()}
            />
            <button
              className="video-browse-btn"
              title="Chọn file video"
              onClick={openFilePicker}
            >📂</button>
            <input
              id="video-file-picker"
              type="file"
              accept="video/*"
              style={{ display: 'none' }}
              onChange={e => {
                const file = e.target.files?.[0]
                if (file) setVideoPath(file.name)
                e.target.value = ''
              }}
            />
          </div>
          <div className="video-action-row">
            <button
              className="video-play-btn"
              onClick={sendVideoPath}
              disabled={wsStatus !== 'connected'}
            >
              ▶ Phát
            </button>
          </div>
        </div>

        <div className="sidebar-footer">
          <span className="footer-label">IP máy chủ</span>
          <span className="footer-ip">{localIP}</span>
          <span className="footer-ip muted">:3001 recognition</span>
        </div>
      </aside>

      {/* ── Stream area ── */}
      <main className="stream-area">
        {selectedCamera && videoEndedMap[selectedCamera] ? (
          <div className="stream-ended">
            <span className="stream-ended-msg">⏹ Video đã kết thúc</span>
            <div className="stream-ended-actions">
              {videoEndedMap[selectedCamera].path && (
                <button
                  className="stream-ended-replay"
                  onClick={() => replayVideo(videoEndedMap[selectedCamera].path)}
                >↺ Phát lại</button>
              )}
              <button
                className="stream-ended-remove"
                onClick={() => removeCamera(selectedCamera)}
              >× Xóa</button>
            </div>
          </div>
        ) : (
          <>
            <img
              ref={imgRef}
              className="stream-img"
              alt="live recognition stream"
              style={{ display: hasFrame ? 'block' : 'none' }}
            />
            {!hasFrame && (
              <div className="stream-placeholder">
                <p>Chờ hình ảnh từ camera...</p>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}

export default App
