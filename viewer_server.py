"""
3D Viewer HTTP Server for ComfyUI

Lightweight HTTP server to serve the 3D viewer web interface and PLY files.
"""

import os
import threading
import webbrowser
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import mimetypes


class ViewerHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for serving viewer files and PLY files from anywhere."""

    def __init__(self, *args, web_dir=None, **kwargs):
        self.web_directory = web_dir
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Special endpoint to serve files from arbitrary paths
        if path == '/file':
            self.serve_file_from_params()
        else:
            # Serve static files from web directory
            super().do_GET()

    def end_headers(self):
        """Add cache-control headers to prevent browser caching."""
        # Disable caching for all files to ensure fresh content
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def serve_file_from_params(self):
        """Serve a file from the path specified in query parameters."""
        try:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            if 'path' not in params:
                self.send_error(400, "Missing 'path' parameter")
                return

            file_path = unquote(params['path'][0])

            # Security: Ensure the file exists and is readable
            if not os.path.isfile(file_path):
                self.send_error(404, f"File not found: {file_path}")
                return

            # Determine content type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                # Default to binary for .ply files
                if file_path.endswith('.ply'):
                    content_type = 'application/octet-stream'
                elif file_path.endswith('.splat'):
                    content_type = 'application/octet-stream'
                else:
                    content_type = 'application/octet-stream'

            # Read and serve the file
            with open(file_path, 'rb') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)

        except Exception as e:
            self.send_error(500, f"Error serving file: {str(e)}")

    def log_message(self, format, *args):
        """Override to provide cleaner logging."""
        # Only log errors and important messages
        if '404' in str(args) or '500' in str(args):
            print(f"[Viewer Server] {format % args}")


class ViewerServer:
    """
    Manages the HTTP server for the 3D viewer.
    Singleton pattern to ensure only one server runs at a time.
    """

    _instance = None
    _server = None
    _server_thread = None
    _port = 8765  # Default port

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ViewerServer, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_web_directory(cls):
        """Get the path to the web directory."""
        current_dir = Path(__file__).parent
        web_dir = current_dir / "web"
        return str(web_dir)

    @classmethod
    def is_running(cls):
        """Check if server is already running."""
        return cls._server is not None and cls._server_thread is not None

    @classmethod
    def start(cls, port=8765):
        """Start the HTTP server if not already running."""
        if cls.is_running():
            print(f"[Viewer Server] Already running on port {cls._port}")
            return cls._port

        cls._port = port
        web_dir = cls.get_web_directory()

        # Ensure web directory exists
        if not os.path.isdir(web_dir):
            raise FileNotFoundError(f"Web directory not found: {web_dir}")

        # Create handler with custom web directory
        def handler(*args, **kwargs):
            return ViewerHTTPHandler(*args, web_dir=web_dir, **kwargs)

        # Try to start server on the specified port, increment if busy
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                cls._server = HTTPServer(('localhost', port), handler)
                cls._port = port
                break
            except OSError as e:
                if attempt < max_attempts - 1:
                    port += 1
                else:
                    raise RuntimeError(f"Failed to start server after {max_attempts} attempts") from e

        # Start server in a daemon thread
        def run_server():
            print(f"[Viewer Server] Starting on http://localhost:{cls._port}")
            print(f"[Viewer Server] Serving files from: {web_dir}")
            cls._server.serve_forever()

        cls._server_thread = threading.Thread(target=run_server, daemon=True)
        cls._server_thread.start()

        # Give server a moment to start
        time.sleep(0.5)

        return cls._port

    @classmethod
    def stop(cls):
        """Stop the HTTP server."""
        if cls._server:
            print("[Viewer Server] Stopping...")
            cls._server.shutdown()
            cls._server.server_close()
            cls._server = None
            cls._server_thread = None
            print("[Viewer Server] Stopped")

    @classmethod
    def get_url(cls):
        """Get the base URL of the running server."""
        if not cls.is_running():
            return None
        return f"http://localhost:{cls._port}"


def open_viewer(file_path, mode="pointcloud", port=8765, auto_open=True):
    """
    Open the 3D viewer for a given file.

    Args:
        file_path (str): Path to the .ply or .splat file
        mode (str): Viewer mode - "splat" or "pointcloud"
        port (int): Port to run the server on (default: 8765)
        auto_open (bool): Whether to automatically open the browser

    Returns:
        str: URL to access the viewer
    """
    # Ensure file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Convert to absolute path
    file_path = os.path.abspath(file_path)

    # Start server if not running
    actual_port = ViewerServer.start(port=port)

    # Build viewer URL
    url = f"http://localhost:{actual_port}/index.html?file={file_path}&mode={mode}"

    print(f"\n{'='*70}")
    print(f"3D Viewer URL: {url}")
    print(f"File: {file_path}")
    print(f"Mode: {mode}")
    print(f"{'='*70}\n")

    # Open in browser
    if auto_open:
        webbrowser.open(url)

    return url


# Convenience function for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python viewer_server.py <path_to_ply_file> [mode]")
        print("  mode: 'splat' or 'pointcloud' (default: pointcloud)")
        sys.exit(1)

    file_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "pointcloud"

    try:
        url = open_viewer(file_path, mode=mode)
        print(f"Viewer opened at: {url}")
        print("Press Ctrl+C to stop the server...")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping server...")
        ViewerServer.stop()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
