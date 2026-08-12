"""Static matplotlib figures for the README and the results directory.

House style: one accent hue per figure, recessive chrome, no chartjunk. Colours
come from a palette validated for colour-vision deficiency, so series stay
distinguishable for readers with protanopia or deuteranopia.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in WSL or CI; render straight to file.

import matplotlib.pyplot as plt
import pandas as pd

from src.data.stats import daily_profile, weekly_profile

# Validated categorical slots; see the palette reference.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BAND = "#9ec5f4"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def style_axes(ax) -> None:
    """Push the chrome into the background so the data reads first."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def plot_eda(
    series: pd.Series,
    output_path: Path | str,
    zoom_start: str | pd.Timestamp | None = None,
    zoom_days: int = 3,
) -> Path:
    """Four-panel exploratory figure for the arrival trace.

    Panels: the whole training series, a minute-level zoom that shows the
    burstiness a smoothed view hides, and the daily and weekly seasonal
    profiles the forecasting models will exploit.

    Args:
        series: 1-minute arrival counts (training portion only).
        output_path: Destination PNG.
        zoom_start: Start of the detail window; defaults to mid-series.
        zoom_days: Width of the detail window in days.

    Returns:
        The path written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(13, 10), facecolor=SURFACE)
    grid = figure.add_gridspec(3, 2, hspace=0.42, wspace=0.18)

    _plot_full_series(figure.add_subplot(grid[0, :]), series)
    _plot_zoom(figure.add_subplot(grid[1, :]), series, zoom_start, zoom_days)
    _plot_daily_profile(figure.add_subplot(grid[2, 0]), series)
    _plot_weekly_profile(figure.add_subplot(grid[2, 1]), series)

    figure.suptitle(
        "NYC taxi pickups as a job-arrival trace (training window)",
        fontsize=14,
        color=INK,
        x=0.5,
        y=0.965,
    )
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return output_path


def _plot_full_series(ax, series: pd.Series) -> None:
    """Whole training window: raw minutes as a noise band, hourly mean on top."""
    hourly = series.resample("1h").mean()

    ax.plot(
        series.index,
        series.to_numpy(),
        color=BAND,
        linewidth=0.3,
        alpha=0.8,
        label="1-minute counts",
    )
    ax.plot(
        hourly.index,
        hourly.to_numpy(),
        color=BLUE,
        linewidth=0.9,
        label="hourly mean",
    )

    style_axes(ax)
    ax.set_title("Full training series", fontsize=11, loc="left")
    ax.set_ylabel("arrivals / min")
    ax.set_xlim(series.index.min(), series.index.max())
    ax.legend(
        loc="upper right", frameon=False, fontsize=8, labelcolor=MUTED, ncols=2
    )


def _plot_zoom(
    ax,
    series: pd.Series,
    zoom_start: str | pd.Timestamp | None,
    zoom_days: int,
) -> None:
    """Minute-level detail.

    This is the panel that justifies the whole project: at 1-minute resolution
    the arrivals are visibly jagged, not a smooth curve. A sine-plus-noise
    generator would produce a tidy band here and make forecasting trivial.
    """
    if zoom_start is None:
        midpoint = len(series) // 2
        zoom_start = series.index[midpoint].normalize()
    zoom_start = pd.Timestamp(zoom_start)
    zoom_end = zoom_start + pd.Timedelta(days=zoom_days)

    window = series.loc[zoom_start:zoom_end]

    ax.plot(window.index, window.to_numpy(), color=BLUE, linewidth=0.7)

    style_axes(ax)
    ax.set_title(
        f"{zoom_days}-day detail at 1-minute resolution", fontsize=11, loc="left"
    )
    ax.set_ylabel("arrivals / min")
    ax.set_xlim(window.index.min(), window.index.max())


def _plot_daily_profile(ax, series: pd.Series) -> None:
    """Mean shape of a day, with the day-to-day spread as a band."""
    mean = daily_profile(series)
    low = daily_profile(series, agg=lambda group: group.quantile(0.1))
    high = daily_profile(series, agg=lambda group: group.quantile(0.9))

    hours = [minute / 60 for minute in mean.index]

    ax.fill_between(
        hours,
        low.to_numpy(),
        high.to_numpy(),
        color=BAND,
        alpha=0.55,
        linewidth=0,
        label="p10-p90 across days",
    )
    ax.plot(hours, mean.to_numpy(), color=BLUE, linewidth=1.4, label="mean")

    style_axes(ax)
    ax.set_title("Daily profile", fontsize=11, loc="left")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("arrivals / min")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=MUTED)


def _plot_weekly_profile(ax, series: pd.Series) -> None:
    """Mean shape of a week, one point per hour, Monday first."""
    profile = weekly_profile(series)

    ax.plot(profile.index, profile.to_numpy(), color=ORANGE, linewidth=1.4)

    style_axes(ax)
    ax.set_title("Weekly profile", fontsize=11, loc="left")
    ax.set_xlabel("day of week")
    ax.set_ylabel("arrivals / min")
    ax.set_xlim(0, 168)
    ax.set_xticks([day * 24 + 12 for day in range(7)])
    ax.set_xticklabels(DAY_NAMES)
    # Hairline separators between days make the shape easier to read across
    # the boundary than tick labels alone.
    for day in range(1, 7):
        ax.axvline(day * 24, color=GRID, linewidth=0.6, zorder=0)
