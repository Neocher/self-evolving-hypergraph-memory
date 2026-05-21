"""Start SHM server in background mode — avoid uvicorn PTY issues"""
import sys, os, signal, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Fork and detach
pid = os.fork()
if pid > 0:
    # Parent exits
    print(f"Server started, PID={pid}")
    sys.exit(0)

# Child continues — detach from terminal
os.setsid()
os.umask(0)

# Redirect stdio
with open('/dev/null', 'r') as f:
    os.dup2(f.fileno(), 0)
with open('/home/admin/shm/shm_new.log', 'a') as f:
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)

# Import and run
from api.app import create_app
import uvicorn

app = create_app()
uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
