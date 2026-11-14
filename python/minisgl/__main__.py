"""``python -m minisgl --model <path>`` starts the HTTP server."""

from .server.launch import launch_server

if __name__ == "__main__":
    launch_server()
