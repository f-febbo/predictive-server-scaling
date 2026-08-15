variable "project" {
  description = "Name prefix for every resource, so teardown is unambiguous."
  type        = string
  default     = "predictive-autoscaler"
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

# --- fleet sizing -----------------------------------------------------------

variable "instance_type" {
  description = "Worker instance type. Small and cheap; the workload is a sleep."
  type        = string
  default     = "t4g.small"
}

variable "asg_max_size" {
  description = <<-EOT
    Hard ceiling on the fleet. This is a safety cap, not a tuning parameter:
    a bug in the scaler cannot cost more than this many instances per hour.
    Keep it low.
  EOT
  type        = number
  default     = 8

  validation {
    condition     = var.asg_max_size <= 50
    error_message = "asg_max_size is capped at 50 to prevent a runaway bill."
  }
}

# NOTE ON EC2 SPOT QUOTAS
#
# The binding constraint on this stack is not cost, it is the account's spot
# vCPU quota. A new account typically allows 32 vCPUs of standard spot, which
# at 2 vCPU per t4g.small is 16 instances TOTAL, shared across both arms.
#
# The first run of this stack was configured well past that ceiling: the
# scaler asked for 16 instances for one arm alone, and the ASG spent half an
# hour failing launches with "Max spot instance count exceeded" while its
# queue backed up. Nothing in the Terraform surfaced the limit, because
# nothing checks it.
#
# So asg_max_size and arrival_divisor have to be chosen together against the
# quota:
#
#     peak instances per arm  ~=  peak_arrivals_per_min
#                                 -------------------------------------
#                                 arrival_divisor x 60 / service_seconds
#                                 x target_utilization
#
# With the trace's 137/min peak, arrival_divisor=15 and a 30s service time,
# that is about 6 instances per arm, or 12 of the 16 available. Check the
# quota before raising either:
#
#     aws service-quotas get-service-quota --service-code ec2 \
#       --quota-code L-34B43A08 --region us-east-1

variable "asg_min_size" {
  description = "Floor on the fleet."
  type        = number
  default     = 1
}

# --- workload ---------------------------------------------------------------

variable "service_seconds" {
  description = <<-EOT
    Seconds a worker spends on each message. Matches the simulator's default so
    that the deployed system and the offline study describe the same workload.
  EOT
  type        = number
  default     = 30
}

variable "app_warmup_seconds" {
  description = <<-EOT
    Delay before a freshly booted worker accepts its first message, modelling
    application warmup. EC2 launch alone is roughly 60-90s; this brings the
    observed end-to-end boot delay near the 180s the simulator assumed.
  EOT
  type        = number
  default     = 90
}

variable "arrival_divisor" {
  description = <<-EOT
    The replayed trace is divided by this before being sent. The point is to
    shrink the fleet to a handful of instances without touching the timescale.

    Time is deliberately NOT compressed. Instance boot time is fixed at two to
    three minutes by physics, so replaying a day in an hour would inflate the
    boot delay to over an hour of trace time and demonstrate a completely
    different regime from the one the simulator studied. Scaling volume keeps
    the ratio of boot delay to demand-change timescale intact, which is the
    only thing that makes the live run comparable to the offline results.
  EOT
  type        = number
  default     = 15
}

variable "horizon_minutes" {
  description = "Forecast horizon. Boot time plus a margin."
  type        = number
  default     = 15
}

variable "forecast_quantile" {
  description = "Which quantile of the forecast to provision against."
  type        = number
  default     = 0.9
}

variable "target_utilization" {
  description = "Fraction of capacity to run at. Below 1.0 buys headroom."
  type        = number
  default     = 0.8
}

# --- comparison arm ---------------------------------------------------------

variable "enable_native_predictive_scaling" {
  description = <<-EOT
    Deploy a second ASG driven by AWS native Predictive Scaling with a custom
    SQS metric specification, as a comparison point.

    Note the limitation: native Predictive Scaling needs at least 24 hours of
    metric history before it emits any forecast, and AWS recommends 14 days for
    a good one. A short run demonstrates the integration but does not give the
    managed feature a fair hearing on forecast quality.
  EOT
  type        = bool
  default     = true
}

# --- guard rails ------------------------------------------------------------

variable "backlog_drain_seconds" {
  description = <<-EOT
    Time budget for the scaler to clear an existing backlog.

    This is the reactive floor under the predictive policy. Without it the
    scaler provisions exactly enough for the incoming rate and nothing to catch
    up with, so any backlog persists indefinitely -- which is precisely what
    happened on the first run when spot capacity became unavailable and the
    fleet could not recover afterwards.
  EOT
  type        = number
  default     = 300
}

variable "shutdown_after_hours" {
  description = <<-EOT
    Hours after the replay starts at which the watchdog pins every fleet to
    zero capacity. The replay itself is 48 hours, so the default leaves a
    couple of hours of slack to observe the drain before the meter stops.

    This is a cost backstop, not a teardown: the stack still has to be
    destroyed with `destroy.sh`.
  EOT
  type        = number
  default     = 50
}

variable "monthly_budget_usd" {
  description = "Budget alarm threshold. Deliberately low."
  type        = number
  default     = 10
}

variable "budget_alert_email" {
  description = "Where budget alerts go. Empty disables the budget."
  type        = string
  default     = ""
}

variable "allowed_ssh_cidr" {
  description = "CIDR permitted to SSH to workers. Empty disables SSH entirely."
  type        = string
  default     = ""
}

variable "instance_types" {
  description = <<-EOT
    Instance types the Auto Scaling groups may launch, best-first. Passed
    through to the arm module; see its variables.tf for the Free Tier and
    architecture constraints that decide what is legal here.
  EOT
  type        = list(string)
  default     = ["t4g.small", "t4g.micro"]
}
