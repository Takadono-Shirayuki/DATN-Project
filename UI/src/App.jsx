import { useState, useEffect, useRef } from 'react'
import './App.css'
import CameraList from './components/CameraList'

function App() {
  const [hasFrame, setHasFrame] = useState(false)
  const [wsStatus, setWsStatus] = useState('connecting')
  const [cameras, setCameras] = useState([])        // list of device IPs seen
  const [selectedCamera, setSelectedCamera] = useState(null)
  const [localIP, setLocalIP] = useState('...')
  const imgRef = useRef(null)
  const wsRef = useRef(null)
  const seenIPs = useRef(new Set())

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
            // Update image via direct DOM (avoids React re-render at 30fps)
            if (imgRef.current) {
              imgRef.current.src = `data:image/jpeg;base64,${msg.frame}`
            }
            if (!hasFrame) setHasFrame(true)

            // Track camera IPs from metadata
            const ip = msg.metadata?.device
            if (ip && !seenIPs.current.has(ip)) {
              seenIPs.current.add(ip)
              setCameras(prev => [...prev, ip])
              setSelectedCamera(prev => prev ?? ip)
            }
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

  const removeCamera = (ip) => {
    seenIPs.current.delete(ip)
    setCameras(prev => prev.filter(c => c !== ip))
    if (selectedCamera === ip) setSelectedCamera(null)
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
          <span className="logo">GR2</span>
          <span className="subtitle">Gait Recognition</span>
          <span className={`ws-badge ws-${wsStatus}`}>{wsLabel}</span>
        </div>

        <CameraList
          cameras={cameras}
          selected={selectedCamera}
          onSelect={setSelectedCamera}
          onRemove={removeCamera}
        />

        <div className="sidebar-footer">
          <span className="footer-label">IP máy chủ</span>
          <span className="footer-ip">{localIP}</span>
          <span className="footer-ip muted">:3001 recognition</span>
        </div>
      </aside>

      {/* ── Stream area ── */}
      <main className="stream-area">
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
      </main>
    </div>
  )
}

export default App
