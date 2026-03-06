function CameraList({ cameras, selected, onSelect, onRemove }) {
  if (cameras.length === 0) {
    return (
      <ul className="camera-list">
        <li className="empty-list">
          Chưa có camera<br />kết nối
        </li>
      </ul>
    )
  }

  return (
    <ul className="camera-list">
      {cameras.map(ip => (
        <li
          key={ip}
          className={`camera-item${selected === ip ? ' active' : ''}`}
          onClick={() => onSelect(ip)}
        >
          <span className="dot" />
          <span className="camera-ip">{ip}</span>
          <button
            className="remove-btn"
            title="Xoá"
            onClick={e => { e.stopPropagation(); onRemove(ip) }}
          >
            ×
          </button>
        </li>
      ))}
    </ul>
  )
}

export default CameraList
