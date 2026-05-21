"""CLI commands for data generation and inspection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from geoplay.data.config import GenerationConfig
from geoplay.data.events import write_events_to_parquet
from geoplay.data.players import generate_players

DEFAULT_OUTPUT_DIR = Path("data/raw")

app = typer.Typer(
    name="data",
    help="Synthetic data generation and inspection.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def generate(
    n_players: int = typer.Option(
        50_000,
        "--players",
        "-p",
        help="Number of players to generate.",
        min=1,
    ),
    n_days: int = typer.Option(
        180,
        "--days",
        "-d",
        help="Number of days of activity to simulate.",
        min=1,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output",
        "-o",
        help="Directory where Parquet files are written.",
    ),
    batch_size: int = typer.Option(
        1_000,
        "--batch-size",
        help="Number of players per Parquet partition.",
        min=1,
    ),
    noise_level: float = typer.Option(
        0.15,
        "--noise",
        help="Stochastic noise level (0.0 to 1.0).",
        min=0.0,
        max=1.0,
    ),
    start_date: str = typer.Option(
        "2025-01-01",
        "--start-date",
        help="Start date of the simulation (YYYY-MM-DD).",
    ),
) -> None:
    """Generate synthetic players and events.

    Writes a `players.parquet` file and a partitioned `events/` directory
    of Parquet files under the output directory.

    Examples
    --------

        # Default scale: 50k players, 180 days
        geoplay data generate

        # Small scale for fast iteration
        geoplay data generate --players 1000 --days 30

        # Custom seed and output
        geoplay data generate --seed 123 --output /tmp/geoplay-data
    """
    parsed_start_date = datetime.strptime(start_date, "%Y-%m-%d")
    config = GenerationConfig(
        n_players=n_players,
        n_days=n_days,
        seed=seed,
        output_dir=output_dir,
        noise_level=noise_level,
        start_date=parsed_start_date,
    )

    _print_config_summary(config)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate players (fast, in-memory).
    with console.status("[bold cyan]Generating players...", spinner="dots"):
        players = generate_players(config)

    players_path = output_dir / "players.parquet"
    players.to_parquet(players_path, engine="pyarrow", index=False)
    console.print(
        f"  [green]✓[/] {len(players):,} players written to "
        f"[cyan]{players_path}[/] ({players_path.stat().st_size / 1024:.0f} KB)"
    )

    # Generate events (slower, streamed to disk).
    n_batches = (len(players) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating events", total=n_batches)
        events_in_progress = {"count": 0}

        def on_batch(batch_idx: int, total: int, n_events: int) -> None:
            events_in_progress["count"] += n_events
            progress.update(
                task,
                advance=1,
                description=f"Generating events ({events_in_progress['count']:,} so far)",
            )

        stats = write_events_to_parquet(
            players_df=players,
            start_date=config.start_date,
            n_days=config.n_days,
            noise_level=config.noise_level,
            output_dir=output_dir,
            seed=config.seed,
            batch_size=batch_size,
            progress_callback=on_batch,
        )

    _print_summary(stats, output_dir)


def _print_config_summary(config: GenerationConfig) -> None:
    """Print a Rich table with the configuration for visibility."""
    table = Table(title="Generation Configuration", show_header=False, padding=(0, 2))
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Players", f"{config.n_players:,}")
    table.add_row("Days", f"{config.n_days}")
    table.add_row("Seed", str(config.seed))
    table.add_row("Geographic center", f"{config.geographic_center}")
    table.add_row("Radius (km)", f"{config.geographic_radius_km}")
    table.add_row("Noise level", f"{config.noise_level}")
    table.add_row("Start date", config.start_date.strftime("%Y-%m-%d"))
    table.add_row("Output directory", str(config.output_dir))
    console.print(table)
    console.print()


def _print_summary(stats: dict[str, int], output_dir: Path) -> None:
    """Print a Rich summary table after generation."""
    events_dir = output_dir / "events"
    total_size_bytes = sum(p.stat().st_size for p in events_dir.glob("*.parquet"))
    total_size_mb = total_size_bytes / (1024 * 1024)

    console.print()
    table = Table(title="Generation Summary", show_header=False, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")
    table.add_row("Total events", f"{stats['total_events']:,}")
    table.add_row("Total sessions", f"{stats['total_sessions']:,}")
    table.add_row("Parquet files", str(stats["n_batches"]))
    table.add_row("Total size", f"{total_size_mb:.1f} MB")
    table.add_row(
        "Avg events/session", f"{stats['total_events'] / max(stats['total_sessions'], 1):.1f}"
    )
    console.print(table)
