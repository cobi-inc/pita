"""
Tests for ValkeyManager utility class.

This module tests the ValkeyManager class that manages the Valkey server subprocess
for inter-process communication during inference.
"""
import pytest
import subprocess
import time
from unittest.mock import patch, MagicMock
import psutil

from pita.utils.valkey_manager import ValkeyManager
from pita.utils.constants import VALKEY_PORT


class TestValkeyManager:
    """Test suite for ValkeyManager class."""
    
    def setup_method(self):
        """Reset ValkeyManager state before each test."""
        # Ensure we start with a clean state
        ValkeyManager._process = None
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Stop any running valkey process that we may have started
        ValkeyManager.stop()
    
    def test_find_valkey_executable_returns_string_or_none(self):
        """Test that _find_valkey_executable returns a string path or None."""
        result = ValkeyManager._find_valkey_executable()
        assert result is None or isinstance(result, str)
        
    def test_is_valkey_running_returns_bool(self):
        """Test that _is_valkey_running returns a boolean (checks for valkey-server or redis-server)."""
        result = ValkeyManager._is_valkey_running()
        assert isinstance(result, bool)
    
    def test_stop_with_no_process_does_not_raise(self):
        """Test that calling stop() when no process is running doesn't raise an error."""
        ValkeyManager._process = None
        ValkeyManager.stop()  # Should not raise
        assert ValkeyManager._process is None
    
    @patch.object(ValkeyManager, '_find_valkey_executable')
    @patch.object(ValkeyManager, '_is_valkey_running')
    def test_start_prints_warning_when_executable_not_found(
        self, mock_is_running, mock_find_executable, capsys
    ):
        """Test that start() prints a warning when valkey-server is not found."""
        mock_is_running.return_value = False
        mock_find_executable.return_value = None
        
        ValkeyManager.start()
        
        captured = capsys.readouterr()
        assert "valkey-server executable not found" in captured.out
        assert ValkeyManager._process is None
    
    @patch.object(ValkeyManager, '_is_valkey_running')
    def test_start_skips_when_already_running(self, mock_is_running):
        """Test that start() skips if Valkey is already running."""
        mock_is_running.return_value = True
        ValkeyManager._process = None
        
        ValkeyManager.start()
        
        # Process should still be None since we detected an existing instance
        assert ValkeyManager._process is None
    
    def test_start_skips_when_process_already_started(self):
        """Test that start() skips if we already started a process."""
        mock_process = MagicMock()
        ValkeyManager._process = mock_process
        
        ValkeyManager.start()
        
        # Process should remain unchanged
        assert ValkeyManager._process is mock_process
    
    @patch('subprocess.Popen')
    @patch.object(ValkeyManager, '_find_valkey_executable')
    @patch.object(ValkeyManager, '_is_valkey_running')
    def test_start_creates_subprocess_with_correct_args(
        self, mock_is_running, mock_find_executable, mock_popen, capsys
    ):
        """Test that start() creates a subprocess with the correct arguments."""
        mock_is_running.return_value = False
        mock_find_executable.return_value = '/usr/bin/valkey-server'
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process running
        mock_popen.return_value = mock_process
        
        ValkeyManager.start()
        
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        
        assert cmd[0] == '/usr/bin/valkey-server'
        assert '--port' in cmd
        assert str(VALKEY_PORT) in cmd
        
        captured = capsys.readouterr()
        assert "Valkey server started successfully" in captured.out
    
    @patch('subprocess.Popen')
    @patch.object(ValkeyManager, '_find_valkey_executable')
    @patch.object(ValkeyManager, '_is_valkey_running')
    def test_start_handles_subprocess_failure(
        self, mock_is_running, mock_find_executable, mock_popen, capsys
    ):
        """Test that start() handles subprocess failure gracefully."""
        mock_is_running.return_value = False
        mock_find_executable.return_value = '/usr/bin/valkey-server'
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process exited
        mock_popen.return_value = mock_process
        
        ValkeyManager.start()
        
        captured = capsys.readouterr()
        assert "Failed to start valkey-server subprocess" in captured.out
        assert ValkeyManager._process is None
    
    def test_stop_terminates_process(self):
        """Test that stop() terminates the managed process."""
        mock_process = MagicMock()
        mock_process.wait = MagicMock()
        ValkeyManager._process = mock_process
        
        ValkeyManager.stop()
        
        mock_process.terminate.assert_called_once()
        assert ValkeyManager._process is None
    
    @patch('subprocess.Popen')
    @patch.object(ValkeyManager, '_find_valkey_executable')
    @patch.object(ValkeyManager, '_is_valkey_running')
    def test_start_handles_popen_exception(
        self, mock_is_running, mock_find_executable, mock_popen, capsys
    ):
        """Test that start() handles exceptions from Popen."""
        mock_is_running.return_value = False
        mock_find_executable.return_value = '/usr/bin/valkey-server'
        mock_popen.side_effect = OSError("Permission denied")
        
        ValkeyManager.start()
        
        captured = capsys.readouterr()
        assert "Failed to start valkey-server" in captured.out
        assert ValkeyManager._process is None


class TestValkeyManagerIntegration:
    """Integration tests for ValkeyManager - requires valkey-server to be installed."""
    
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup before and after each test."""
        ValkeyManager.stop()
        yield
        ValkeyManager.stop()
    
    @pytest.mark.skipif(
        ValkeyManager._find_valkey_executable() is None,
        reason="valkey-server not installed"
    )
    def test_start_and_stop_integration(self):
        """Integration test: start and stop the Valkey server (or detect existing)."""
        # Check if a compatible server is already running
        server_was_running = ValkeyManager._is_valkey_running()
        
        ValkeyManager.start()
        
        # Give it a moment to fully start
        time.sleep(0.5)
        
        if server_was_running:
            # If a server was already running, start() should have detected it
            # and not started a new process
            assert ValkeyManager._process is None, \
                "ValkeyManager should not start a new process when server already running"
        else:
            # If no server was running, start() should have started one
            assert ValkeyManager._process is not None, \
                "ValkeyManager should have started a new process"
            assert ValkeyManager._process.poll() is None, \
                "Started process should still be running"
        
        ValkeyManager.stop()
        
        # After stop, our managed process should be cleaned up
        assert ValkeyManager._process is None
