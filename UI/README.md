# Gait Recognition - UI Server

## Tổng quan

Giao diện Server dùng React + Vite để:
1. Quản lý đăng ký các ứng dụng camera
2. Hiển thị real-time stream từ các camera
3. Xem lịch sử và kết quả nhận diện

## Cấu trúc dự án

```
UI/
├── src/
│   ├── components/
│   │   ├── CameraStream.jsx
│   │   ├── CameraList.jsx
│   │   ├── Registration.jsx
│   │   ├── GaitAnalysis.jsx
│   │   └── Dashboard.jsx
│   ├── pages/
│   │   ├── Home.jsx
│   │   ├── CameraDetail.jsx
│   │   └── Results.jsx
│   ├── hooks/
│   │   ├── useWebSocket.js
│   │   └── useCamera.js
│   ├── services/
│   │   ├── api.js
│   │   └── websocket.js
│   ├── styles/
│   │   └── App.css
│   ├── App.jsx
│   └── main.jsx
├── public/
├── vite.config.js
├── package.json
└── README.md
```

## Tính năng chính

### 1. Đăng ký Camera
- Form đăng ký ứng dụng camera mới
- Lưu IP, tên thiết bị
- Assign device ID

### 2. Hiển thị Stream
- Real-time MJPEG stream từ camera
- Canvas rendering
- WebSocket cho live update

### 3. Dashboard
- Danh sách camera kết nối
- Trạng thái online/offline
- Xem stream từ camera

### 4. Gait Analysis View (Tương lai)
- Hiển thị GEI
- Kết quả phân loại hành động
- Lịch sử nhận dện

## Công nghệ

- **React 18+** - UI library
- **Vite** - Build tool (nhanh, hiện đại)
- **JavaScript/JSX** - Frontend logic
- **WebSocket** - Real-time communication
- **Tailwind CSS hoặc Material-UI** - Styling

## Cài đặt

```bash
npm install
```

## Chạy Development

```bash
npm run dev
```

## Build Production

```bash
npm run build
```

## Backend Integration

- **WebSocket**: `ws://localhost:3000/stream`
  - Nhận frames từ camera
  - Gửi commands đến camera app

- **REST API** (tương lai):
  - `POST /api/camera/register` - Đăng ký camera
  - `GET /api/camera` - Danh sách camera
  - `GET /api/camera/:id/stream` - Stream

## TODO

- [ ] Thiết lập Vite + React
- [ ] Layout cơ bản
- [ ] Component CameraList
- [ ] Component Registration
- [ ] Component CameraStream
- [ ] WebSocket integration
- [ ] Real-time display
- [ ] Dashboard
- [ ] API integration
