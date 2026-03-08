"""
Flask-based web UI for remote control of the GR2 Camera system.

Features:
  - Password-authenticated login (same password as the GUI)
  - Dashboard: current mode/status, switch modes, trigger action
  - Settings: change server IP/port, change password
  - JSON endpoint: GET /api/state (polled by the dashboard JS)

Runs in a daemon background thread started by main.py.
"""

import ipaddress
import json
import threading
import time

import auth
import config
from app_controller import AppController, MODES

try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    from flask import (
        Flask, Response, render_template_string, request,
        session, redirect, url_for, jsonify,
    )
    _FLASK_OK = True
except ImportError:
    _FLASK_OK = False

# ---------------------------------------------------------------------------
# HTML templates (inline â€” no external files needed for embedded deployment)
# ---------------------------------------------------------------------------

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>GR2 Camera â€“ Login</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;
         display:flex;align-items:center;justify-content:center;min-height:100vh}
    .card{background:#1e293b;border-radius:12px;padding:40px;min-width:340px}
    h1{color:#4ade80;margin-bottom:24px;font-size:1.4rem;text-align:center}
    label{display:block;margin-bottom:6px;font-size:13px;color:#94a3b8}
    input[type=password]{width:100%;padding:10px 14px;border-radius:6px;
      border:1px solid #334155;background:#0f172a;color:#e2e8f0;font-size:15px;
      margin-bottom:16px;outline:none}
    input[type=password]:focus{border-color:#4ade80}
    button{width:100%;padding:12px;background:#4ade80;color:#0f172a;
           border:none;border-radius:6px;font-size:15px;font-weight:bold;cursor:pointer}
    button:hover{background:#22c55e}
    .err{color:#f87171;margin-bottom:12px;font-size:14px;text-align:center}
  </style>
</head>
<body>
  <div class="card">
    <h1>&#127909; GR2 Camera</h1>
    {% if error %}<p class="err">{{ error }}</p>{% endif %}
    <form method="post">
      <label>Password</label>
      <input type="password" name="password" autofocus autocomplete="current-password">
      <button type="submit">Login</button>
    </form>
  </div>
</body>
</html>"""

_DASH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>GR2 Camera</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,sans-serif;background:#000;color:#e2e8f0;
         display:flex;flex-direction:column;height:100vh;overflow:hidden}
    #hdr{background:#1c1c1e;padding:10px 16px;display:flex;
         justify-content:space-between;align-items:center;
         border-bottom:1px solid #2c2c2e;flex-shrink:0}
    #hdr h1{color:#4ade80;font-size:1rem}
    #hdr a{color:#94a3b8;text-decoration:none;font-size:13px}
    #hdr a:hover{color:#e2e8f0}
    #layout{display:flex;flex:1;overflow:hidden}
    /* Left sidebar */
    #sidebar{width:90px;background:#1c1c1e;display:flex;
             flex-direction:column;align-items:center;
             padding:12px 0;border-right:1px solid #2c2c2e;flex-shrink:0}
    .mode-btn{width:78px;padding:10px 4px;border:none;background:transparent;
              color:#94a3b8;cursor:pointer;border-radius:8px;
              display:flex;flex-direction:column;align-items:center;
              gap:4px;margin-bottom:4px}
    .mode-btn:hover{background:#2c2c2e;color:#e2e8f0}
    .mode-btn.active{background:rgba(10,132,255,.15);color:#0a84ff}
    .mode-btn .icon{font-size:22px;line-height:1}
    .mode-btn .lbl{font-size:10px;font-weight:bold;text-transform:uppercase;
                   letter-spacing:.06em}
    .sb-div{width:60px;border:none;border-top:1px solid #2c2c2e;margin:6px 0}
    .sb-info{font-size:10px;color:#4b5563;text-align:center;padding:4px 8px}
    /* Camera area */
    #cam-area{flex:1;position:relative;background:#000;
              display:flex;align-items:center;justify-content:center;
              overflow:hidden}
    #feed{max-width:100%;max-height:100%;object-fit:contain;display:block}
    #cam-overlay{position:absolute;inset:0;background:rgba(0,0,0,.8);
                 display:none;align-items:center;justify-content:center;
                 color:#93c5fd;font-size:1.1rem;text-align:center;
                 white-space:pre-line;line-height:1.6;pointer-events:none}
    #sbar{position:absolute;bottom:0;left:0;right:0;
          background:rgba(0,0,0,.55);color:#94a3b8;
          font-size:12px;padding:4px 12px;
          display:flex;justify-content:space-between;align-items:center}
    .r-dot{font-weight:bold;font-size:11px}
    .dot-rec{color:#ef4444}
    .dot-live{color:#4ade80}
    .dot-off{display:none}
    /* Right panel */
    #rpanel{width:220px;background:#1e293b;display:flex;
            flex-direction:column;border-left:1px solid #2c2c2e;flex-shrink:0}
    #settings-body{flex:1;overflow-y:auto;padding:14px 12px}
    .sec-hdr{font-size:10px;font-weight:bold;text-transform:uppercase;
             letter-spacing:.08em;color:#64748b;margin-bottom:10px;
             padding-bottom:6px;border-bottom:1px solid #334155}
    .fg{margin-bottom:10px}
    label.fl{display:block;font-size:12px;color:#94a3b8;margin-bottom:3px}
    input[type=text],input[type=number],input[type=password]{
      width:100%;padding:7px 10px;background:#0f172a;
      border:1px solid #334155;border-radius:5px;
      color:#e2e8f0;font-size:13px;outline:none}
    input:focus{border-color:#4ade80}
    .btn-save{padding:7px 14px;background:#4ade80;color:#0f172a;border:none;
              border-radius:5px;cursor:pointer;font-weight:bold;font-size:13px;
              margin-top:4px}
    .btn-save:hover{background:#22c55e}
    .msg{font-size:12px;margin-top:6px}
    .msg-ok{color:#4ade80}
    .msg-err{color:#f87171}
    .rp-div{border:none;border-top:1px solid #2c2c2e;margin:12px 0}
    /* Action button */
    #action-wrap{padding:10px 12px;flex-shrink:0}
    #action-btn{width:100%;padding:20px 10px;border:none;border-radius:10px;
                cursor:pointer;font-size:13px;font-weight:bold;
                display:flex;flex-direction:column;align-items:center;
                gap:6px;background:#4CAF50;color:#fff}
    #action-btn .icon{font-size:28px;line-height:1}
    #action-btn:hover{filter:brightness(1.12)}
    #action-btn:active{filter:brightness(.9)}
  </style>
</head>
<body>
  <div id="hdr">
    <h1>&#127909; GR2 Camera</h1>
    <a href="/logout">Logout</a>
  </div>
  <div id="layout">

    <!-- Left sidebar: mode buttons -->
    <div id="sidebar">
      <button class="mode-btn" id="btn-photo" onclick="setMode('photo')">
        <span class="icon">&#128247;</span>
        <span class="lbl">Photo</span>
      </button>
      <button class="mode-btn" id="btn-video" onclick="setMode('video')">
        <span class="icon">&#127916;</span>
        <span class="lbl">Video</span>
      </button>
      <button class="mode-btn" id="btn-stream" onclick="setMode('stream')">
        <span class="icon">&#128225;</span>
        <span class="lbl">Stream</span>
      </button>
      <hr class="sb-div">
      <div class="sb-info" id="fps-info" style="margin-top:auto">-- FPS</div>
    </div>

    <!-- Camera feed -->
    <div id="cam-area">
      <img id="feed" src="/video_feed" alt="camera">
      <div id="cam-overlay">
        <span id="overlay-txt">&#128225; Connecting...</span>
      </div>
      <div id="sbar">
        <span id="status-txt">{{ state.status }}</span>
        <span>
          <span id="rec-dot" class="r-dot dot-rec{{ '' if state.recording else ' dot-off' }}">&#9679; REC</span>&nbsp;
          <span id="live-dot" class="r-dot dot-live{{ '' if state.streaming else ' dot-off' }}">&#9679; LIVE</span>&nbsp;
          <span id="fps-bar">{{ state.fps }} FPS</span>
        </span>
      </div>
    </div>

    <!-- Right panel: settings + action button -->
    <div id="rpanel">
      <div id="settings-body">

        <div class="sec-hdr">Server</div>
        <form method="post" action="/api/settings/server">
          <div class="fg">
            <label class="fl">IP Address</label>
            <input type="text" name="ip" value="{{ server_ip }}" placeholder="192.168.1.100">
          </div>
          <div class="fg">
            <label class="fl">Port</label>
            <input type="number" name="port" value="{{ server_port }}" min="1" max="65535">
          </div>
          <button class="btn-save" type="submit">Save</button>
          {% if srv_msg %}
          <div class="msg {{ 'msg-ok' if srv_ok else 'msg-err' }}">{{ srv_msg }}</div>
          {% endif %}
        </form>

        <hr class="rp-div">

        <div class="sec-hdr">Password</div>
        <form method="post" action="/api/settings/password">
          <div class="fg">
            <label class="fl">Current</label>
            <input type="password" name="old_password" autocomplete="current-password">
          </div>
          <div class="fg">
            <label class="fl">New</label>
            <input type="password" name="new_password" autocomplete="new-password">
          </div>
          <div class="fg">
            <label class="fl">Confirm</label>
            <input type="password" name="confirm_password" autocomplete="new-password">
          </div>
          <button class="btn-save" type="submit">Change</button>
          {% if pw_msg %}
          <div class="msg {{ 'msg-ok' if pw_ok else 'msg-err' }}">{{ pw_msg }}</div>
          {% endif %}
        </form>

        {% if flash_msg %}
        <hr class="rp-div">
        <div class="msg {{ 'msg-ok' if flash_ok else 'msg-err' }}">{{ flash_msg }}</div>
        {% endif %}

      </div>

      <!-- Action button -->
      <div id="action-wrap">
        <button id="action-btn" onclick="doAction()">
          <span class="icon"></span>
          <span id="action-lbl"></span>
        </button>
      </div>
    </div>

  </div>
  <script>
    var _cdTimer = null;
    function setMode(m){
      var fd=new FormData();fd.append('mode',m);
      fetch('/api/mode',{method:'POST',body:fd,headers:{'X-Ajax':'1'}})
        .then(function(){setTimeout(refreshState,200)});
    }
    function doAction(){
      fetch('/api/action',{method:'POST',headers:{'X-Ajax':'1'}})
        .then(function(){setTimeout(refreshState,200)});
    }
    function refreshState(){
      fetch('/api/state').then(function(r){return r.json();})
        .then(applyState).catch(function(){});
    }
    function startCountdown(n, ip, port){
      var ov=document.getElementById('cam-overlay');
      ov.style.display='flex';
      document.getElementById('overlay-txt').innerHTML=
        '\u26A0 Connection failed<br><br>'+ip+':'+port+'<br><br>'+
        'Switching to Photo mode in '+n+'s...';
      if(n<=0){
        _cdTimer=null;
        setMode('photo');
        return;
      }
      _cdTimer=setTimeout(function(){startCountdown(n-1,ip,port);},1000);
    }
    function applyState(d){
      ['photo','video','stream'].forEach(function(m){
        var b=document.getElementById('btn-'+m);
        if(b)b.className='mode-btn'+(d.mode===m?' active':'');
      });
      document.getElementById('fps-info').textContent=d.fps+' FPS';
      document.getElementById('fps-bar').textContent=d.fps+' FPS';
      document.getElementById('status-txt').textContent=d.status;
      document.getElementById('rec-dot').className='r-dot dot-rec'+(d.recording?'':' dot-off');
      document.getElementById('live-dot').className='r-dot dot-live'+(d.streaming?'':' dot-off');
      var btn=document.getElementById('action-btn');
      var icn=btn.querySelector('.icon');
      var lbl=document.getElementById('action-lbl');
      if(d.mode==='photo'){
        btn.style.background='#4CAF50';icn.textContent='📸';lbl.textContent='Capture';
      }else if(d.mode==='video'){
        if(d.recording){btn.style.background='#F44336';icn.textContent='\u23F9';lbl.textContent='Stop';}
        else{btn.style.background='#607D8B';icn.textContent='\u23FA';lbl.textContent='Record';}
      }else if(d.mode==='stream'){
        if(d.streaming){btn.style.background='#FF5722';icn.textContent='\u23F9';lbl.textContent='Stop';}
        else{btn.style.background='#2196F3';icn.textContent='\u25B6';lbl.textContent='Stream';}
      }
      var ov=document.getElementById('cam-overlay');
      if(d.mode==='stream'&&!d.streaming){
        ov.style.display='flex';
        if(d.stream_failed){
          // Start countdown only once; don't reset if already running
          if(!_cdTimer){
            var addr=(d.stream_fail_addr||'?:?');
            var parts=addr.split(':');
            var ip=parts[0]||'?', port=parts[1]||'?';
            startCountdown(5,ip,port);
          }
        }else{
          // Still connecting — clear any stale countdown
          if(_cdTimer){clearTimeout(_cdTimer);_cdTimer=null;}
          document.getElementById('overlay-txt').textContent='Connecting...';
        }
      }else{
        ov.style.display='none';
        if(_cdTimer){clearTimeout(_cdTimer);_cdTimer=null;}
      }
    }
    setInterval(refreshState,2000);
    window.onload=function(){
      applyState({{ state|tojson }});
      refreshState();
    };
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_flask_app(ctrl: AppController) -> "Flask":
    """Build and return the Flask WSGI application."""
    cfg_data = config.load()
    app = Flask(__name__)
    app.secret_key = bytes.fromhex(cfg_data["web"]["secret_key"])

    # ---- auth decorator ------------------------------------------------
    from functools import wraps

    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("authenticated"):
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated

    # ---- routes --------------------------------------------------------

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            pw = request.form.get("password", "")
            if auth.verify(pw):
                session["authenticated"] = True
                session.permanent = False
                return redirect(url_for("dashboard"))
            error = "Incorrect password."
        return render_template_string(_LOGIN_HTML, error=error)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/")
    @login_required
    def dashboard():
        state = ctrl.get_state()
        ip, port = config.get_server()
        return render_template_string(
            _DASH_HTML,
            state=state,
            server_ip=ip,
            server_port=port,
            flash_msg=request.args.get("flash", ""),
            flash_ok=request.args.get("flash_ok", "0") == "1",
            srv_msg=request.args.get("srv_msg", ""),
            srv_ok=request.args.get("srv_ok", "0") == "1",
            pw_msg=request.args.get("pw_msg", ""),
            pw_ok=request.args.get("pw_ok", "0") == "1",
        )

    @app.route("/api/state")
    @login_required
    def api_state():
        return jsonify(ctrl.get_state())

    @app.route("/api/mode", methods=["POST"])
    @login_required
    def api_mode():
        mode = request.form.get("mode", "")
        if mode in MODES:
            ctrl.enqueue("switch_mode", mode)
        if request.headers.get("X-Ajax"):
            return jsonify({"ok": True})
        return redirect(url_for("dashboard"))

    @app.route("/api/action", methods=["POST"])
    @login_required
    def api_action():
        ctrl.enqueue("action")
        if request.headers.get("X-Ajax"):
            return jsonify({"ok": True})
        return redirect(url_for("dashboard", flash="Command sent", flash_ok=1))

    @app.route("/api/settings/server", methods=["POST"])
    @login_required
    def api_server():
        ip = request.form.get("ip", "").strip()
        port_str = request.form.get("port", "3001").strip()
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            return redirect(url_for("dashboard", srv_msg="Invalid IP address", srv_ok=0))
        try:
            port = int(port_str)
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            return redirect(url_for("dashboard", srv_msg="Invalid port number", srv_ok=0))
        config.set_server(ip, port)
        return redirect(url_for("dashboard", srv_msg="Server settings saved", srv_ok=1))

    @app.route("/api/settings/password", methods=["POST"])
    @login_required
    def api_password():
        old = request.form.get("old_password", "")
        new = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        if new != confirm:
            return redirect(url_for("dashboard", pw_msg="Passwords do not match", pw_ok=0))
        if len(new) < 4:
            return redirect(url_for("dashboard", pw_msg="Password too short (min 4)", pw_ok=0))
        if auth.change_password(old, new):
            session.clear()          # force re-login after password change
            return redirect(url_for("login"))
        return redirect(url_for("dashboard", pw_msg="Current password incorrect", pw_ok=0))

    @app.route("/video_feed")
    @login_required
    def video_feed():
        if not _CV2_OK:
            return "cv2 not available", 503

        def _gen():
            while True:
                frame = ctrl.get_frame()
                if frame is None:
                    time.sleep(0.033)
                    continue
                ok, buf = _cv2.imencode(
                    ".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if not ok:
                    time.sleep(0.033)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                time.sleep(0.033)

        return Response(
            _gen(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    return app


# ---------------------------------------------------------------------------
# Entry point called from main.py
# ---------------------------------------------------------------------------

def start_web_server(ctrl: AppController, port: int) -> None:
    """Start the Flask web server in a daemon background thread."""
    if not _FLASK_OK:
        print("[WebServer] Flask not installed â€” web UI disabled.")
        print("[WebServer] Install with: pip install flask")
        return

    app = create_flask_app(ctrl)

    def _run():
        # Suppress Flask's default startup banner
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True, name="web-server")
    t.start()
    print(f"[WebServer] Listening on http://0.0.0.0:{port}/")
