"""Pull the live AWS run out of CloudWatch before the stack is destroyed.

Run with:

    uv run python -m experiments.collect_live_run

Produces:
    results/phase5_live_run.csv   5-minute samples of both arms
    results/phase5_live_run.md    summary table

Once `destroy.sh` runs, the queues and Auto Scaling groups are gone and the
resource names with them. CloudWatch keeps the metrics for months, but nothing
else records what the run was, so this script has to be run first. The CSV it
writes is the committed record.

The live run is not the source of the project's headline numbers -- those come
from the simulator, where a run is free, repeatable, and not at the mercy of
spot capacity. This is evidence the control loop closes on real infrastructure.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import boto3
import pandas as pd

RESULTS_DIR = Path("results")
REGION = "us-east-1"
PROJECT = "predictive-autoscaler"

# The replay window, from the Terraform output at deploy time.
RUN_START = dt.datetime(2026, 8, 12, 20, 47, 2, tzinfo=dt.timezone.utc)
RUN_END = dt.datetime(2026, 8, 14, 20, 47, 2, tzinfo=dt.timezone.utc)

# Mid-run, three faults were fixed: the scaler gained the reactive floor it
# should have shipped with, the group gained instance-type diversity after
# running out of t4g.small spot capacity, and message retention was raised so
# the age metric stopped truncating. Everything before this point measures
# those faults rather than the scaling strategies.
FIXES_APPLIED = dt.datetime(2026, 8, 14, 1, 11, 0, tzinfo=dt.timezone.utc)

# The fixes did not take effect instantly: the backlog built up beforehand took
# until roughly 02:00 to drain, and until it did, the age metric still reflected
# messages that had been queued for hours under the broken configuration. The
# usable window starts once that has cleared.
BACKLOG_CLEARED = dt.datetime(2026, 8, 14, 2, 0, 0, tzinfo=dt.timezone.utc)

# From here to the end of the replay, both groups were largely unable to launch
# instances at all, for two reasons that have nothing to do with scaling
# strategy: genuine spot capacity shortage, and -- worse -- the instance-type
# list added to work around that shortage. Six of its seven entries were not
# free-tier-eligible, and this is a Free Tier account, so every attempt to
# launch them failed instantly. 146 failed launches on one group, 74 on the
# other. Capacity, not policy, is what these hours measure.
DEGRADATION_START = dt.datetime(2026, 8, 14, 12, 0, 0, tzinfo=dt.timezone.utc)

SLO_SECONDS = 60.0
PERIOD = 300  # 5 minutes; a 48h window at 1 minute exceeds the API's limit

ARMS = ("custom", "native")


def fetch(client, namespace, metric, dimension, value, stat) -> pd.Series:
    response = client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric,
        Dimensions=[{"Name": dimension, "Value": value}],
        StartTime=RUN_START,
        EndTime=RUN_END,
        Period=PERIOD,
        Statistics=[stat],
    )
    points = response.get("Datapoints", [])
    if not points:
        return pd.Series(dtype=float)

    frame = pd.DataFrame(points).sort_values("Timestamp")
    return pd.Series(
        frame[stat].to_numpy(),
        index=pd.to_datetime(frame["Timestamp"], utc=True),
    )


def collect() -> pd.DataFrame:
    client = boto3.client("cloudwatch", region_name=REGION)
    columns = {}

    for arm in ARMS:
        queue = f"{PROJECT}-{arm}-work"
        group = f"{PROJECT}-{arm}-asg"

        columns[f"{arm}_age_s"] = fetch(
            client, "AWS/SQS", "ApproximateAgeOfOldestMessage", "QueueName", queue, "Maximum"
        )
        columns[f"{arm}_depth"] = fetch(
            client, "AWS/SQS", "ApproximateNumberOfMessagesVisible", "QueueName", queue, "Average"
        )
        columns[f"{arm}_sent"] = fetch(
            client, "AWS/SQS", "NumberOfMessagesSent", "QueueName", queue, "Sum"
        )
        columns[f"{arm}_instances"] = fetch(
            client, "AWS/AutoScaling", "GroupInServiceInstances",
            "AutoScalingGroupName", group, "Average",
        )

    return pd.DataFrame(columns).sort_index()


def _stats(window: pd.DataFrame, arm: str) -> dict:
    age = window[f"{arm}_age_s"].dropna()
    instances = window[f"{arm}_instances"].dropna()
    return {
        "sent": window[f"{arm}_sent"].sum(),
        "violation": (age > SLO_SECONDS).mean() if len(age) else float("nan"),
        "p50": age.median() if len(age) else float("nan"),
        "p99": age.quantile(0.99) if len(age) else float("nan"),
        "max": age.max() if len(age) else float("nan"),
        "mean_instances": instances.mean() if len(instances) else float("nan"),
    }


def _table(window: pd.DataFrame) -> list[str]:
    row = {arm: _stats(window, arm) for arm in ARMS}

    def line(label, key, fmt):
        return f"| {label} | {fmt(row['custom'][key])} | {fmt(row['native'][key])} |"

    return [
        "| | custom (forecast) | native (AWS) |",
        "|---|---|---|",
        line("Messages enqueued", "sent", lambda v: f"{v:,.0f}"),
        line(f"Samples over {SLO_SECONDS:g}s", "violation", lambda v: f"{v:.1%}"),
        line("Median age of oldest", "p50", lambda v: f"{v:.0f}s"),
        line("p99 age of oldest", "p99", lambda v: f"{v:.0f}s"),
        line("Worst age of oldest", "max", lambda v: f"{v:.0f}s"),
        line("Mean instances", "mean_instances", lambda v: f"{v:.1f}"),
    ]


def summarise(samples: pd.DataFrame) -> str:
    usable = samples[
        (samples.index >= BACKLOG_CLEARED) & (samples.index < DEGRADATION_START)
    ]
    degraded = samples[samples.index >= DEGRADATION_START]

    return "\n".join(
        [
            "# Live AWS run",
            "",
            f"Replay window: {RUN_START:%Y-%m-%d %H:%M} to {RUN_END:%Y-%m-%d %H:%M} UTC, "
            f"sampled every {PERIOD // 60} minutes.",
            "",
            "Two identical worker fleets fed identical load. The only difference "
            "is what sets capacity: a forecasting Lambda on one, AWS native "
            "Predictive Scaling plus target tracking on the other.",
            "",
            "## What this run does and does not show",
            "",
            "It shows the control loop closing on real infrastructure: the "
            "scaler read live CloudWatch metrics, applied a forecast, converted "
            "it to a fleet size, and drove a real Auto Scaling group every "
            "minute for two days.",
            "",
            "It does **not** produce a usable comparison between the two "
            "strategies. Of 48 hours, roughly 10 were clean. The rest were "
            "dominated by faults and by the environment:",
            "",
            f"- **Before {FIXES_APPLIED:%H:%M} on {FIXES_APPLIED:%b %d}** — three of my own "
            "bugs. The scaler shipped without the reactive floor the simulation "
            "explicitly recommended, so it could not work off a backlog. The "
            "group offered one spot instance type and could not launch when that "
            "ran out. Message retention was short enough that the age metric "
            "truncated at exactly one hour while older messages were deleted.",
            f"- **After {DEGRADATION_START:%H:%M} on {DEGRADATION_START:%b %d}** — neither group "
            "could launch instances. Partly genuine spot shortage, but mostly a "
            "worse bug: the instance-type list I added to survive that shortage "
            "contained six types that are not free-tier-eligible, and this is a "
            "Free Tier account. Every launch of them failed instantly. 146 "
            "failed launches on one group, 74 on the other. These hours measure "
            "AWS capacity and my configuration, not scaling policy.",
            "- Both fleets were capped at 8 instances by the spot vCPU quota, so "
            "one instance is a large fraction of the fleet and capacity moves in "
            "coarse steps.",
            "- AWS Predictive Scaling needs 24 hours of history before it "
            "forecasts at all, and recommends 14 days. Over two days it is "
            "effectively target tracking alone. This is not a fair test of the "
            "managed feature and is not presented as one.",
            "",
            "The project's headline numbers come from the simulator, where a run "
            "is free, repeatable, and not at the mercy of spot capacity. That "
            "was the design from the start, and this run is a good argument for "
            "it.",
            "",
            f"## The usable window ({BACKLOG_CLEARED:%H:%M}-{DEGRADATION_START:%H:%M}, "
            f"{len(usable) * PERIOD / 3600:.0f} hours)",
            "",
            "Both arms healthy and keeping up. In this window the AWS-scaled "
            "arm edged the forecasting one -- fewer samples over the threshold, "
            "on slightly fewer instances.",
            "",
            "That is not evidence the forecast lost. Ten hours at about four "
            "messages a minute, against a fleet the spot quota capped at eight "
            "and a floor of one, is far too small and far too coarse to "
            "separate two policies: a single instance is a third of the fleet, "
            "and a single stuck message moves the age metric more than any "
            "scaling decision does. It is reported because cropping it would be "
            "dishonest, not because it means anything.",
            "",
            *_table(usable),
            "",
            "Median age sits near the 60s threshold for both arms because of "
            "something the simulator does not model: a real worker long-polls "
            "SQS for up to 20 seconds before it even sees a message, and then "
            "takes 30 seconds to process it. The simulator dispatches instantly, "
            "so its 60s target is far easier to meet than the same number is "
            "here. The live and simulated violation rates are not comparable, "
            "and the SLO threshold should have been recalibrated for the "
            "deployment.",
            "",
            f"## The degraded window ({DEGRADATION_START:%H:%M} onward), for completeness",
            "",
            *_table(degraded),
            "",
            "Recorded so the failure is visible rather than cropped. Both arms "
            "look terrible, and neither number says anything about scaling.",
        ]
    ) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    samples = collect()
    if samples.empty:
        raise SystemExit("no metrics returned; has the stack already been destroyed?")

    samples.to_csv(RESULTS_DIR / "phase5_live_run.csv")
    report = summarise(samples)
    (RESULTS_DIR / "phase5_live_run.md").write_text(report, encoding="utf-8")

    print(f"{len(samples)} samples written to {RESULTS_DIR / 'phase5_live_run.csv'}\n")
    print(report)


if __name__ == "__main__":
    main()
