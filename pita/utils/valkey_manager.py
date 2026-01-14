"""
Manager for starting and stopping a Valkey server subprocess.

This module provides the ValkeyManager class that automatically manages a Valkey
server instance for use by the PITA package. It can detect existing Valkey instances
and start a new one if needed.
"""
import subprocess
import atexit
import time
import shutil
import psutil
import os
from pita.utils.constants import VALKEY_PORT
from typing import Optional


class ValkeyManager:
    """
    Manager for starting and stopping a Valkey server subprocess.

    This class provides utilities to automatically manage a Valkey server instance
    for use by the PITA package. It can detect existing Valkey instances and start
    a new one if needed.

    Attributes:
        _process: The subprocess.Popen instance for the managed Valkey server, or None.
    """
    _process: Optional[subprocess.Popen] = None

    @classmethod
    def start(cls) -> None:
        """Starts the Valkey server if it is not already running."""
        # Check if we already started it
        if cls._process is not None:
            return

        # Check if already running system-wide or by another process
        if cls._is_valkey_running():
            return

        valkey_executable = cls._find_valkey_executable()
        
        if not valkey_executable:
            print("Warning: valkey-server executable not found. Please ensure it is installed.")
            return

        print(f"Starting valkey-server on port {VALKEY_PORT}...")
        try:
            # Start valkey-server as a subprocess
            # We use --daemonize no so we can control the subprocess lifecycle easily
            cls._process = subprocess.Popen(
                [valkey_executable, '--port', str(VALKEY_PORT), '--save', '', '--appendonly', 'no'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait a moment to ensure it starts
            time.sleep(0.5)
            if cls._process.poll() is not None:
                print("Failed to start valkey-server subprocess.")
                cls._process = None
            else:
                print("Valkey server started successfully.")
                # Register cleanup on exit
                atexit.register(cls.stop)
                
        except Exception as e:
            print(f"Failed to start valkey-server: {e}")

    @classmethod
    def stop(cls) -> None:
        """
        Stop the Valkey server if it was started by this manager.

        This method attempts to gracefully terminate the Valkey process, waiting up to
        2 seconds before forcefully killing it if necessary.
        """
        if cls._process:
            print("Stopping valkey-server...")
            cls._process.terminate()
            try:
                cls._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                cls._process.kill()
            cls._process = None

    @classmethod
    def _is_valkey_running(cls) -> bool:
        """
        Check if a Valkey or Redis server process is currently running on the system.

        Since Valkey and Redis are protocol-compatible, we check for both server types
        to avoid starting a new instance when an existing compatible server is running.

        Returns:
            True if a valkey-server or redis-server process is found, False otherwise.
        """
        for proc in psutil.process_iter(['name']):
            try:
                proc_name = proc.info['name']
                if 'valkey-server' in proc_name or 'redis-server' in proc_name:
                    # Could add stricter check for port/args, but simplified for now
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have terminated or be inaccessible; safely skip it and continue scanning
                pass
        return False
        
    @classmethod
    def _find_valkey_executable(cls) -> Optional[str]:
        """
        Find the valkey-server executable in various system locations.

        This method searches for valkey-server in the system PATH, conda environments,
        and relative to the Python executable.

        Returns:
            The absolute path to the valkey-server executable, or None if not found.
        """
        import sys
        executable = shutil.which('valkey-server')
        if executable:
            return executable
            
        # Fallback to checking conda env bin if not in PATH
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            candidate = os.path.join(conda_prefix, 'bin', 'valkey-server')
            if os.path.exists(candidate):
                return candidate
        
        # Fallback relative to sys.executable (common in conda envs)
        # sys.executable is .../bin/python, so we look in .../bin/valkey-server
        bin_dir = os.path.dirname(sys.executable)
        candidate = os.path.join(bin_dir, 'valkey-server')
        if os.path.exists(candidate):
            return candidate

        return None
