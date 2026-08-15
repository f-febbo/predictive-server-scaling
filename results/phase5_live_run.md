# Live AWS run

Replay window: 2026-08-12 20:47 to 2026-08-14 20:47 UTC, sampled every 5 minutes.

Two identical worker fleets fed identical load. The only difference is what sets capacity: a forecasting Lambda on one, AWS native Predictive Scaling plus target tracking on the other.

## What this run does and does not show

It shows the control loop closing on real infrastructure: the scaler read live CloudWatch metrics, applied a forecast, converted it to a fleet size, and drove a real Auto Scaling group every minute for two days.

It does **not** produce a usable comparison between the two strategies. Of 48 hours, roughly 10 were clean. The rest were dominated by faults and by the environment:

- **Before 01:11 on Aug 14** — three of my own bugs. The scaler shipped without the reactive floor the simulation explicitly recommended, so it could not work off a backlog. The group offered one spot instance type and could not launch when that ran out. Message retention was short enough that the age metric truncated at exactly one hour while older messages were deleted.
- **After 12:00 on Aug 14** — neither group could launch instances. Partly genuine spot shortage, but mostly a worse bug: the instance-type list I added to survive that shortage contained six types that are not free-tier-eligible, and this is a Free Tier account. Every launch of them failed instantly. 146 failed launches on one group, 74 on the other. These hours measure AWS capacity and my configuration, not scaling policy.
- Both fleets were capped at 8 instances by the spot vCPU quota, so one instance is a large fraction of the fleet and capacity moves in coarse steps.
- AWS Predictive Scaling needs 24 hours of history before it forecasts at all, and recommends 14 days. Over two days it is effectively target tracking alone. This is not a fair test of the managed feature and is not presented as one.

The project's headline numbers come from the simulator, where a run is free, repeatable, and not at the mercy of spot capacity. That was the design from the start, and this run is a good argument for it.

## The usable window (02:00-12:00, 10 hours)

Both arms healthy and keeping up. In this window the AWS-scaled arm edged the forecasting one -- fewer samples over the threshold, on slightly fewer instances.

That is not evidence the forecast lost. Ten hours at about four messages a minute, against a fleet the spot quota capped at eight and a floor of one, is far too small and far too coarse to separate two policies: a single instance is a third of the fleet, and a single stuck message moves the age metric more than any scaling decision does. It is reported because cropping it would be dishonest, not because it means anything.

| | custom (forecast) | native (AWS) |
|---|---|---|
| Messages enqueued | 2,218 | 2,218 |
| Samples over 60s | 27.5% | 12.5% |
| Median age of oldest | 52s | 56s |
| p99 age of oldest | 230s | 298s |
| Worst age of oldest | 406s | 359s |
| Mean instances | 3.4 | 3.0 |

Median age sits near the 60s threshold for both arms because of something the simulator does not model: a real worker long-polls SQS for up to 20 seconds before it even sees a message, and then takes 30 seconds to process it. The simulator dispatches instantly, so its 60s target is far easier to meet than the same number is here. The live and simulated violation rates are not comparable, and the SLO threshold should have been recalibrated for the deployment.

## The degraded window (12:00 onward), for completeness

| | custom (forecast) | native (AWS) |
|---|---|---|
| Messages enqueued | 2,540 | 2,540 |
| Samples over 60s | 71.4% | 85.7% |
| Median age of oldest | 1970s | 5816s |
| p99 age of oldest | 7317s | 13072s |
| Worst age of oldest | 7429s | 13138s |
| Mean instances | 3.6 | 3.4 |

Recorded so the failure is visible rather than cropped. Both arms look terrible, and neither number says anything about scaling.
