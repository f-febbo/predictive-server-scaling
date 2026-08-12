# Live deployment

A minimal, cheap deployment that closes the loop end to end on real
infrastructure: real SQS queues, real spot instances with real boot delays, a
real scaler running once a minute. It exists to show the control loop working
outside the simulator, **not** to reproduce the offline results. All of the
measurement in this project happens in the simulator, where a run is free and
repeatable; nothing here is iterated against.

## What gets built

Two identical arms, differing only in what decides fleet size:

| Arm | Scaling driven by |
|---|---|
| `custom` | A Lambda that runs every minute, reads the forecast, and calls `SetDesiredCapacity` |
| `native` | AWS native Predictive Scaling on a custom SQS metric, paired with target tracking |

Everything else is shared and identical — same queue configuration, same worker
AMI and script, same instance type, and the same load generator sending the
same messages to both queues in the same invocation. If any of that differed,
the comparison would not be measuring what it claims to.

```
                    ┌──────────────────────────┐
  EventBridge ─1m──►│  load generator Lambda   │
                    └────────────┬─────────────┘
                                 │ same messages to both
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
        ┌────────────────┐              ┌────────────────┐
        │ SQS (custom)   │              │ SQS (native)   │
        └───────┬────────┘              └───────┬────────┘
                │                               │
        ┌───────▼────────┐              ┌───────▼────────┐
        │ ASG (spot)     │              │ ASG (spot)     │
        └───────▲────────┘              └───────▲────────┘
                │ SetDesiredCapacity            │
        ┌───────┴────────┐              ┌───────┴──────────────┐
        │ scaler Lambda  │              │ AWS Predictive       │
        │ ◄─1m─ EventBr. │              │ Scaling + target trk │
        └────────────────┘              └──────────────────────┘
```

## Cost

**Roughly $4–7 for a 48-hour run** in `us-east-1`. Approximate, and prices
change — verify before trusting these.

| Component | 48h | Notes |
|---|---|---|
| EC2 spot, both arms | $3–4 | ~8 instances/arm average, `t4g.small` at ~$0.005/hr spot |
| EBS, 8 GB per instance | ~$0.70 | Deleted with the instance |
| SQS | $0–0.15 | ~350k requests; usually inside the free tier |
| Lambda | ~$0 | 5,760 invocations total, well inside the free tier |
| CloudWatch | ~$0.20 | 4 custom metrics; ASG group metrics and the first 3 dashboards are free |

Things that were deliberately designed out, because they are how a demo stack
quietly runs up a bill:

- **No NAT Gateway.** About $32/month whether or not traffic flows. Workers sit
  in a public subnet and reach SQS over the internet gateway instead.
- **No EC2 detailed monitoring.** About $2.10 per instance-month, and on a
  churning fleet that would rival the compute cost. Nothing here reads
  per-instance metrics — ASG group metrics are one-minute and free.
- **Long polling on SQS** (20s). Short polling bills an empty receive every few
  hundred milliseconds per idle worker.
- **Three-day log retention**, so log ingestion cannot accumulate.
- **A hard `asg_max_size` cap**, validated at 50 or below. This bounds the
  hourly burn rate no matter what the scaler does.

## Running it

Prerequisites: Terraform ≥ 1.5, AWS credentials, and the Phase 4 backtest run
so the forecast table exists.

```bash
# 1. Generate the trace slice and forecast table the Lambdas ship with
uv run python -m experiments.phase4_backtest      # if not already run
uv run python -m experiments.export_infra_payload

# 2. Deploy
cd infra
terraform init
terraform apply

# 3. Watch it
open "$(terraform output -raw dashboard_url)"

# 4. Tear down — do not skip this
./destroy.sh
```

Set a budget alarm before walking away:

```bash
terraform apply -var 'budget_alert_email=you@example.com'
```

`destroy.sh` scales both fleets to zero, waits for instances to drain, runs
`terraform destroy`, and then queries the account for surviving instances and
ASGs. It exits non-zero if anything is still running, so a failed teardown is
loud rather than silent.

## Two things to know before reading any live numbers

**1. Time is not compressed, volume is.** The obvious way to run a cheap demo
is to replay a day in an hour. That does not work here. Instance boot time is
fixed at two to three minutes by physics, so a 24× compression would leave the
boot delay spanning more than an hour of trace time — a completely different
regime from the one the simulator studied, and the boot delay is the entire
subject of this project. Instead the replay runs at 1:1 and divides the arrival
counts by `arrival_divisor` (default 5), which shrinks the fleet to single
digits while preserving the ratio of boot delay to demand-change timescale.

**2. A short run does not give AWS Predictive Scaling a fair hearing.** Native
Predictive Scaling requires at least 24 hours of metric history before it emits
any forecast, and AWS recommends 14 days. Over 48 hours it is working from
almost nothing. A short run demonstrates that the integration is correct and
that the loop closes; it is **not** evidence about the managed feature's
forecast quality, and this repository does not claim otherwise. The structural
argument against it for this workload — hourly forecast granularity, which
cannot address a burst that arrives and clears inside an hour — stands on its
own and does not depend on the live run.

## Where the forecast comes from

The scaler Lambda ships with a lookup table of the LightGBM quantile model's
genuine out-of-sample forecasts, produced by the Phase 4 rolling-origin
backtest: the value stored for minute `m` was computed only from data at or
before `m`. Since the live run replays exactly that historical period,
re-evaluating the model in Lambda would recompute numbers that already exist,
at the price of dragging numpy and scipy into the deployment package for about
60 MB.

The scaler is not open-loop, though. Every invocation reads the real
`NumberOfMessagesSent` metric from CloudWatch and re-levels the forecast by how
the observed rate compares with the trace's expectation, clipped to [0.5, 2.0].
If the load generator stalls or SQS throttles, the fleet follows what is
actually arriving rather than what the script said should arrive.

In a system replaying live traffic rather than a known trace, this handler
would call the model instead of indexing it. The surrounding control loop —
read metrics, forecast, convert via Little's Law, set capacity — is identical
either way, and that loop is what this deployment exists to demonstrate.

## Keeping the deployed maths honest

The Lambda cannot import from `src/`, since it ships as a standalone zip, so
the Little's Law conversion is written out a second time in
`lambda/scaler_handler.py`. `tests/test_infra_lambda.py` asserts the two agree
across a grid of arrival rates, service times, and utilisation targets. Without
that test, the live fleet could quietly be sized by different rules than every
number in the README.
