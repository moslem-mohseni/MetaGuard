"""
MetaGuard CLI

Author: Moslem Mohseni

Command-line interface for fraud detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .. import SimpleDetector, __version__, analyze_transaction_risk, check_transaction
from ..utils.exceptions import InvalidTransactionError, MetaGuardError

# Create Typer app
app = typer.Typer(
    name="metaguard",
    help="MetaGuard - Fraud detection for metaverse transactions",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"MetaGuard version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """MetaGuard - Fraud detection for metaverse transactions."""
    pass


@app.command()
def detect(
    amount: float = typer.Option(..., "--amount", "-a", help="Transaction amount"),
    hour: int = typer.Option(..., "--hour", "-h", help="Hour of day (0-23)"),
    user_age: int = typer.Option(..., "--user-age", "-u", help="Account age in days"),
    tx_count: int = typer.Option(..., "--tx-count", "-t", help="Number of prior transactions"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Detect fraud in a single transaction."""
    try:
        transaction = {
            "amount": amount,
            "hour": hour,
            "user_age_days": user_age,
            "transaction_count": tx_count,
        }

        result = check_transaction(transaction)

        if json_output:
            console.print_json(json.dumps(result))
        else:
            # Create display
            status = "[red]SUSPICIOUS[/red]" if result["is_suspicious"] else "[green]OK[/green]"
            risk_color = {
                "Low": "green",
                "Medium": "yellow",
                "High": "red",
            }.get(result["risk_level"], "white")

            panel = Panel(
                f"Status: {status}\n"
                f"Risk Score: {result['risk_score']:.2%}\n"
                f"Risk Level: [{risk_color}]{result['risk_level']}[/{risk_color}]",
                title="Detection Result",
                border_style="blue",
            )
            console.print(panel)

    except InvalidTransactionError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except MetaGuardError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def analyze(
    amount: float = typer.Option(..., "--amount", "-a", help="Transaction amount"),
    hour: int = typer.Option(..., "--hour", "-h", help="Hour of day (0-23)"),
    user_age: int = typer.Option(..., "--user-age", "-u", help="Account age in days"),
    tx_count: int = typer.Option(..., "--tx-count", "-t", help="Number of prior transactions"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Perform detailed risk analysis on a transaction."""
    try:
        transaction = {
            "amount": amount,
            "hour": hour,
            "user_age_days": user_age,
            "transaction_count": tx_count,
        }

        result = analyze_transaction_risk(transaction)

        if json_output:
            console.print_json(json.dumps(result))
        else:
            # Risk level color
            risk_color = {
                "Low": "green",
                "Medium": "yellow",
                "High": "red",
            }.get(result["risk_level"], "white")

            # Create table for factors
            table = Table(title="Risk Factors")
            table.add_column("Factor", style="cyan")
            table.add_column("Status", justify="center")

            for factor, active in result["factors"].items():
                status = "[red]Active[/red]" if active else "[green]Inactive[/green]"
                table.add_row(factor.replace("_", " ").title(), status)

            console.print(f"\n[bold]Risk Score:[/bold] {result['risk_score']:.1f}/100")
            console.print(
                f"[bold]Risk Level:[/bold] [{risk_color}]{result['risk_level']}[/{risk_color}]"
            )
            console.print(f"[bold]Active Factors:[/bold] {result['active_factor_count']}/4\n")
            console.print(table)

    except InvalidTransactionError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except MetaGuardError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def batch(
    input_file: Path = typer.Argument(..., help="Input JSON file with transactions"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output JSON file for results"
    ),
) -> None:
    """Process multiple transactions from a JSON file."""
    try:
        # Read input file
        if not input_file.exists():
            console.print(f"[red]Error:[/red] File not found: {input_file}")
            raise typer.Exit(code=1)

        with input_file.open() as f:
            data = json.load(f)

        # Handle both list and dict with 'transactions' key
        if isinstance(data, dict):
            transactions = data.get("transactions", [])
        else:
            transactions = data

        if not transactions:
            console.print("[yellow]Warning:[/yellow] No transactions to process")
            raise typer.Exit(code=0)

        # Process transactions
        detector = SimpleDetector()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Processing {len(transactions)} transactions...", total=None)
            results = detector.batch_detect(transactions)

        # Count results
        suspicious_count = sum(1 for r in results if r["is_suspicious"])

        # Display summary
        table = Table(title="Batch Processing Results")
        table.add_column("#", style="dim", width=4)
        table.add_column("Amount", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Risk Level", justify="center")
        table.add_column("Score", justify="right")

        for i, (tx, result) in enumerate(zip(transactions, results), 1):
            status = "[red]SUSPICIOUS[/red]" if result["is_suspicious"] else "[green]OK[/green]"
            risk_color = {"Low": "green", "Medium": "yellow", "High": "red"}.get(
                result["risk_level"], "white"
            )
            table.add_row(
                str(i),
                f"${tx.get('amount', 0):,.2f}",
                status,
                f"[{risk_color}]{result['risk_level']}[/{risk_color}]",
                f"{result['risk_score']:.2%}",
            )

        console.print(table)
        console.print(
            f"\n[bold]Summary:[/bold] {suspicious_count}/{len(results)} suspicious "
            f"({suspicious_count/len(results)*100:.1f}%)"
        )

        # Save results if output file specified
        if output_file:
            output_data = {
                "total": len(results),
                "suspicious": suspicious_count,
                "results": results,
            }
            with output_file.open("w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"\n[green]Results saved to:[/green] {output_file}")

    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in {input_file}: {e}")
        raise typer.Exit(code=1)
    except MetaGuardError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Show model and configuration information."""
    try:
        detector = SimpleDetector()
        info = detector.get_model_info()

        table = Table(title="MetaGuard Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Version", __version__)
        table.add_row("Model Type", info.get("model_type", "Unknown"))
        table.add_row("Model Path", str(info.get("model_path", "Default")))
        table.add_row("Risk Threshold", f"{info.get('risk_threshold', 0.5):.1%}")

        console.print(table)

    except MetaGuardError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),  # nosec B104
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Start the REST API server."""
    try:
        import uvicorn

        console.print(f"[green]Starting MetaGuard API server on {host}:{port}[/green]")
        console.print(f"[dim]API docs: http://{host}:{port}/docs[/dim]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        uvicorn.run(
            "metaguard.api.rest:app",
            host=host,
            port=port,
            reload=reload,
        )
    except ImportError:
        console.print(
            "[red]Error:[/red] uvicorn not installed. " "Install with: pip install metaguard[api]"
        )
        raise typer.Exit(code=1)


def cli_main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    cli_main()
