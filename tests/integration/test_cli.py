"""
CLI Integration Tests

Author: Moslem Mohseni

Tests for Typer CLI commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Skip all tests if typer is not installed
typer = pytest.importorskip("typer")

from typer.testing import CliRunner

from metaguard.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


class TestVersionCommand:
    """Tests for version command."""

    def test_version_flag(self, runner: CliRunner) -> None:
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "MetaGuard version" in result.output

    def test_version_short_flag(self, runner: CliRunner) -> None:
        """Test -v flag."""
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "MetaGuard version" in result.output


class TestDetectCommand:
    """Tests for detect command."""

    def test_detect_basic(self, runner: CliRunner) -> None:
        """Test basic detection."""
        result = runner.invoke(
            app,
            [
                "detect",
                "--amount", "100",
                "--hour", "14",
                "--user-age", "30",
                "--tx-count", "5",
            ],
        )
        assert result.exit_code == 0
        assert "Detection Result" in result.output or "Status:" in result.output

    def test_detect_suspicious(self, runner: CliRunner) -> None:
        """Test detection of suspicious transaction."""
        result = runner.invoke(
            app,
            [
                "detect",
                "--amount", "5000",
                "--hour", "3",
                "--user-age", "2",
                "--tx-count", "50",
            ],
        )
        assert result.exit_code == 0
        assert "High" in result.output or "SUSPICIOUS" in result.output

    def test_detect_json_output(self, runner: CliRunner) -> None:
        """Test JSON output format."""
        result = runner.invoke(
            app,
            [
                "detect",
                "--amount", "100",
                "--hour", "14",
                "--user-age", "30",
                "--tx-count", "5",
                "--json",
            ],
        )
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert "is_suspicious" in data
        assert "risk_score" in data
        assert "risk_level" in data

    def test_detect_short_flags(self, runner: CliRunner) -> None:
        """Test short flag options."""
        result = runner.invoke(
            app,
            [
                "detect",
                "-a", "100",
                "-h", "14",
                "-u", "30",
                "-t", "5",
            ],
        )
        assert result.exit_code == 0

    def test_detect_missing_required(self, runner: CliRunner) -> None:
        """Test missing required options."""
        result = runner.invoke(app, ["detect", "--amount", "100"])
        assert result.exit_code != 0


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_analyze_basic(self, runner: CliRunner) -> None:
        """Test basic analysis."""
        result = runner.invoke(
            app,
            [
                "analyze",
                "--amount", "100",
                "--hour", "14",
                "--user-age", "30",
                "--tx-count", "5",
            ],
        )
        assert result.exit_code == 0
        assert "Risk Score:" in result.output or "Risk Level:" in result.output

    def test_analyze_shows_factors(self, runner: CliRunner) -> None:
        """Test that factors are shown."""
        result = runner.invoke(
            app,
            [
                "analyze",
                "--amount", "5000",
                "--hour", "3",
                "--user-age", "5",
                "--tx-count", "50",
            ],
        )
        assert result.exit_code == 0
        assert "Risk Factors" in result.output or "Active" in result.output

    def test_analyze_json_output(self, runner: CliRunner) -> None:
        """Test JSON output format."""
        result = runner.invoke(
            app,
            [
                "analyze",
                "--amount", "100",
                "--hour", "14",
                "--user-age", "30",
                "--tx-count", "5",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "risk_score" in data
        assert "factors" in data
        assert "active_factor_count" in data


class TestBatchCommand:
    """Tests for batch command."""

    def test_batch_processing(
        self, runner: CliRunner, temp_dir: Path, batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Test batch processing from file."""
        # Create input file
        input_file = temp_dir / "transactions.json"
        input_file.write_text(json.dumps(batch_transactions))

        result = runner.invoke(app, ["batch", str(input_file)])
        assert result.exit_code == 0
        assert "Batch Processing Results" in result.output or "Summary:" in result.output

    def test_batch_with_output_file(
        self, runner: CliRunner, temp_dir: Path, batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Test batch processing with output file."""
        input_file = temp_dir / "transactions.json"
        output_file = temp_dir / "results.json"
        input_file.write_text(json.dumps(batch_transactions))

        result = runner.invoke(
            app, ["batch", str(input_file), "--output", str(output_file)]
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify output file content
        output_data = json.loads(output_file.read_text())
        assert "total" in output_data
        assert "suspicious" in output_data
        assert "results" in output_data

    def test_batch_file_not_found(self, runner: CliRunner) -> None:
        """Test batch with non-existent file."""
        result = runner.invoke(app, ["batch", "nonexistent.json"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_batch_invalid_json(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test batch with invalid JSON."""
        input_file = temp_dir / "invalid.json"
        input_file.write_text("not valid json")

        result = runner.invoke(app, ["batch", str(input_file)])
        assert result.exit_code != 0

    def test_batch_with_transactions_key(
        self, runner: CliRunner, temp_dir: Path, batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Test batch with 'transactions' key in JSON."""
        input_file = temp_dir / "transactions.json"
        input_file.write_text(json.dumps({"transactions": batch_transactions}))

        result = runner.invoke(app, ["batch", str(input_file)])
        assert result.exit_code == 0


class TestInfoCommand:
    """Tests for info command."""

    def test_info_basic(self, runner: CliRunner) -> None:
        """Test info command."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "MetaGuard Information" in result.output or "Model Type" in result.output

    def test_info_shows_version(self, runner: CliRunner) -> None:
        """Test that info shows version."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Version" in result.output or "1." in result.output


class TestHelpCommands:
    """Tests for help output."""

    def test_main_help(self, runner: CliRunner) -> None:
        """Test main help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "MetaGuard" in result.output
        assert "detect" in result.output
        assert "analyze" in result.output
        assert "batch" in result.output
        assert "info" in result.output

    def test_detect_help(self, runner: CliRunner) -> None:
        """Test detect command help."""
        result = runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0
        assert "--amount" in result.output
        assert "--hour" in result.output

    def test_analyze_help(self, runner: CliRunner) -> None:
        """Test analyze command help."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "risk analysis" in result.output.lower() or "analyze" in result.output.lower()

    def test_batch_help(self, runner: CliRunner) -> None:
        """Test batch command help."""
        result = runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output


class TestCLIEdgeCases:
    """Tests for CLI edge cases."""

    def test_detect_large_amount(self, runner: CliRunner) -> None:
        """Test detection with large amount."""
        result = runner.invoke(
            app,
            [
                "detect",
                "--amount", "9999999",
                "--hour", "12",
                "--user-age", "100",
                "--tx-count", "5",
            ],
        )
        assert result.exit_code == 0

    def test_detect_boundary_hour(self, runner: CliRunner) -> None:
        """Test detection with boundary hour values."""
        # Hour 0
        result = runner.invoke(
            app,
            ["detect", "-a", "100", "-h", "0", "-u", "30", "-t", "5"],
        )
        assert result.exit_code == 0

        # Hour 23
        result = runner.invoke(
            app,
            ["detect", "-a", "100", "-h", "23", "-u", "30", "-t", "5"],
        )
        assert result.exit_code == 0

    def test_detect_minimum_values(self, runner: CliRunner) -> None:
        """Test detection with minimum valid values."""
        result = runner.invoke(
            app,
            [
                "detect",
                "--amount", "0.01",
                "--hour", "0",
                "--user-age", "1",
                "--tx-count", "0",
            ],
        )
        assert result.exit_code == 0
