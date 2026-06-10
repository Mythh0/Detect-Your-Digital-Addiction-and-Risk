import http.server, socketserver, os, webbrowser, threading
os.chdir(os.path.join(os.path.dirname(__file__), "frontend"))
PORT = 8080
threading.Thread(target=lambda:[__import__('time').sleep(0.8),webbrowser.open(f"http://localhost:{PORT}")],daemon=True).start()
print(f"✅ Dashboard at http://localhost:{PORT}  (Ctrl+C to stop)")
with socketserver.TCPServer(("",PORT),http.server.SimpleHTTPRequestHandler) as s: s.serve_forever()
