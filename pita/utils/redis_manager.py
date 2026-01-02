import subprocess
import atexit
import time
import shutil
import psutil
import os
from pita.utils.constants import REDIS_PORT
from typing import Optional

class RedisManager:
    """
    Manager for starting and stopping a Redis server subprocess.

    This class provides utilities to automatically manage a Redis server instance
    for use by the PITA package. It can detect existing Redis instances and start
    a new one if needed.

    Attributes:
        _process: The subprocess.Popen instance for the managed Redis server, or None.
    """
    _process: Optional[subprocess.Popen] = None

    @classmethod
    def start(cls) -> None:
        """Starts the Redis server if it is not already running."""
        # Check if we already started it
        if cls._process is not None:
            return

        # Check if already running system-wide or by another process
        if cls._is_redis_running():
            return

        redis_executable = cls._find_redis_executable()
        
        if not redis_executable:
            print("Warning: redis-server executable not found. Please ensure it is installed.")
            return

        print(f"Starting redis-server on port {REDIS_PORT}...")
        try:
            # Start redis-server as a subprocess
            # We use --daemonize no so we can control the subprocess lifecycle easily
            cls._process = subprocess.Popen(
                [redis_executable, '--port', str(REDIS_PORT), '--save', '', '--appendonly', 'no'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait a moment to ensure it starts
            time.sleep(0.5)
            if cls._process.poll() is not None:
                print("Failed to start redis-server subprocess.")
                cls._process = None
            else:
                print("Redis server started successfully.")
                # Register cleanup on exit
                atexit.register(cls.stop)
                
        except Exception as e:
            print(f"Failed to start redis-server: {e}")

    @classmethod
    def stop(cls) -> None:
        """
        Stop the Redis server if it was started by this manager.

        This method attempts to gracefully terminate the Redis process, waiting up to
        2 seconds before forcefully killing it if necessary.
        """
        if cls._process:
            print("Stopping redis-server...")
            cls._process.terminate()
            try:
                cls._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                cls._process.kill()
            cls._process = None

    @classmethod
    def _is_redis_running(cls) -> bool:
        """
        Check if a Redis server process is currently running on the system.

        Returns:
            True if a redis-server process is found, False otherwise.
        """
        for proc in psutil.process_iter(['name']):
            try:
                if 'redis-server' in proc.info['name']:
                    # Could add stricter check for port/args, but simplified for now
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have terminated or be inaccessible; safely skip it and continue scanning
                pass
        return False
        
    @classmethod
    def _find_redis_executable(cls) -> Optional[str]:
        """
        Find the redis-server executable in various system locations.

        This method searches for redis-server in the system PATH, conda environments,
        and relative to the Python executable.

        Returns:
            The absolute path to the redis-server executable, or None if not found.
        """
        import sys
        executable = shutil.which('redis-server')
        if executable:
            return executable
            
        # Fallback to checking conda env bin if not in PATH
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            candidate = os.path.join(conda_prefix, 'bin', 'redis-server')
            if os.path.exists(candidate):
                return candidate
        
        # Fallback relative to sys.executable (common in conda envs)
        # sys.executable is .../bin/python, so we look in .../bin/redis-server
        bin_dir = os.path.dirname(sys.executable)
        candidate = os.path.join(bin_dir, 'redis-server')
        if os.path.exists(candidate):
            return candidate

        return None
