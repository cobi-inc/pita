"""
Tests for the PITA CLI.

This module tests the command-line interface provided by the pita package.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock


class TestCLIGroup:
    """Tests for the main CLI group and help."""
    
    def test_cli_help_shows_version(self):
        """Test that --version flag works."""
        from pita.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert 'pita' in result.output.lower()
        assert '0.0.1' in result.output
    
    def test_cli_help_shows_commands(self):
        """Test that --help shows available commands."""
        from pita.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'serve' in result.output


class TestServeCommand:
    """Tests for the serve subcommand."""
    
    def test_serve_help_shows_options(self):
        """Test that serve --help shows all options."""
        from pita.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['serve', '--help'])
        assert result.exit_code == 0
        assert '--model' in result.output
        assert '--engine' in result.output
        assert '--tokenizer' in result.output
        assert '--port' in result.output
        assert '--host' in result.output
    
    def test_serve_calls_start_server_with_defaults(self):
        """Test that serve command calls start_server with None values (uses defaults)."""
        from pita.cli import cli
        runner = CliRunner()
        
        with patch('pita.api.serve.start_server') as mock_start:
            result = runner.invoke(cli, ['serve'])
            mock_start.assert_called_once_with(
                model=None,
                engine=None,
                tokenizer=None,
                port=None,
                host=None
            )
    
    def test_serve_passes_custom_options(self):
        """Test that custom options are passed to start_server."""
        from pita.cli import cli
        runner = CliRunner()
        
        with patch('pita.api.serve.start_server') as mock_start:
            result = runner.invoke(cli, [
                'serve',
                '--model', 'my-model',
                '--engine', 'llama_cpp',
                '--tokenizer', '/path/to/tokenizer',
                '--port', '9000',
                '--host', '127.0.0.1'
            ])
            mock_start.assert_called_once_with(
                model='my-model',
                engine='llama_cpp',
                tokenizer='/path/to/tokenizer',
                port=9000,
                host='127.0.0.1'
            )
    
    def test_serve_short_options(self):
        """Test that short option flags work."""
        from pita.cli import cli
        runner = CliRunner()
        
        with patch('pita.api.serve.start_server') as mock_start:
            result = runner.invoke(cli, [
                'serve',
                '-m', 'test-model',
                '-e', 'vllm',
                '-p', '8888'
            ])
            mock_start.assert_called_once()
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs['model'] == 'test-model'
            assert call_kwargs['engine'] == 'vllm'
            assert call_kwargs['port'] == 8888
    
    def test_serve_invalid_engine_rejected(self):
        """Test that invalid engine choice is rejected."""
        from pita.cli import cli
        runner = CliRunner()
        
        result = runner.invoke(cli, ['serve', '--engine', 'invalid_engine'])
        assert result.exit_code != 0
        assert 'invalid' in result.output.lower() or 'choice' in result.output.lower()


class TestStartServer:
    """Tests for the start_server function."""
    
    def test_start_server_uses_defaults_when_none(self):
        """Test that start_server uses defaults from environment when args are None."""
        from pita.api.serve import start_server, get_default_config
        
        with patch('pita.api.serve.create_app') as mock_create, \
             patch('pita.api.serve.uvicorn.run') as mock_run:
            
            mock_app = MagicMock()
            mock_create.return_value = mock_app
            
            start_server()
            
            # Verify create_app was called with default config values
            mock_create.assert_called_once()
            config = mock_create.call_args[0][0]
            defaults = get_default_config()
            
            assert config['model'] == defaults['model']
            assert config['engine'] == defaults['engine']
            assert config['port'] == defaults['port']
            assert config['host'] == defaults['host']
    
    def test_start_server_uses_provided_values(self):
        """Test that start_server uses provided values over defaults."""
        from pita.api.serve import start_server
        
        with patch('pita.api.serve.create_app') as mock_create, \
             patch('pita.api.serve.uvicorn.run') as mock_run:
            
            mock_app = MagicMock()
            mock_create.return_value = mock_app
            
            start_server(
                model='custom-model',
                engine='llama_cpp',
                tokenizer='/custom/tokenizer',
                port=9999,
                host='192.168.1.1'
            )
            
            config = mock_create.call_args[0][0]
            assert config['model'] == 'custom-model'
            assert config['engine'] == 'llama_cpp'
            assert config['tokenizer'] == '/custom/tokenizer'
            assert config['port'] == 9999
            assert config['host'] == '192.168.1.1'
            
            mock_run.assert_called_once_with(mock_app, host='192.168.1.1', port=9999)
