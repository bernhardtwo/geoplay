"""GeoPlay CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from geoplay.cli import data as data_commands

app = typer.Typer(
    name="geoplay",
    help="GeoPlay: Geo-contextual player segmentation and content recommendation.",
    no_args_is_help=True,
)
app.add_typer(data_commands.app, name="data")
console = Console()


@app.command()
def version() -> None:
    """Print the installed GeoPlay version."""
    from importlib.metadata import version as get_version

    console.print(f"[bold cyan]geoplay[/] version [green]{get_version('geoplay')}[/]")


@app.command()
def info() -> None:
    """Print project info and roadmap status."""
    console.print("[bold cyan]GeoPlay[/] — geo-contextual recommender")
    console.print("Status: data generation phase.")


if __name__ == "__main__":
    app()
