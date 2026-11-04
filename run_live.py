"""
AI Sprint Coach — 100% Free Live Public Launcher
Runs the local FastAPI backend and generates a secure Cloudflare HTTPS link.
Requires NO credit card, NO signup, and has NO spend limits.
"""

import os
import sys
import re
import time
import subprocess
import threading

def run_uvicorn():
    """Runs the FastAPI server locally."""
    cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cloudflared_bin = os.path.join(base_dir, "cloudflared.exe")

    if not os.path.exists(cloudflared_bin):
        print("[!] cloudflared.exe not found. Downloading Cloudflare Tunnel binary...")
        import urllib.request
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        urllib.request.urlretrieve(url, cloudflared_bin)
        print("[✓] Download complete.\n")

    print("=" * 64)
    print("      AI SPRINT COACH — 100% FREE LIVE HOSTING")
    print("=" * 64)
    print("\n[1/2] Starting local AI pipeline server on port 8000...")
    
    server_thread = threading.Thread(target=run_uvicorn, daemon=True)
    server_thread.start()
    time.sleep(2)

    print("[2/2] Connecting to Cloudflare's global edge network...")
    tunnel_cmd = [cloudflared_bin, "tunnel", "--url", "http://127.0.0.1:8000"]
    
    proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    tunnel_url = None
    for line in proc.stderr:
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            tunnel_url = match.group(0)
            print("\n" + "#" * 64)
            print("  YOUR SPRINT COACH IS LIVE FOR ANYONE TO USE!")
            print(f"  Live URL: {tunnel_url}")
            print("#" * 64 + "\n")
            print("👉 Open that URL on your phone or send it to anyone.")
            print("👉 Live 3D Pose, Gait Analysis & AI Coaching are active.")
            print("👉 Keep this terminal open while sharing (Press Ctrl+C to stop).\n")
            break

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping server and tunnel...")
        proc.terminate()

if __name__ == "__main__":
    main()
