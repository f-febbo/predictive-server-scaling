# One dashboard showing both arms on the same axes.
#
# The panels are ordered the way the argument runs: what arrived, how long the
# oldest message waited (the SLI), how much capacity was in service, and what
# the forecast asked for.

locals {
  arm_names = keys(module.arm)

  age_metrics = [
    for name in local.arm_names : [
      "AWS/SQS", "ApproximateAgeOfOldestMessage",
      "QueueName", module.arm[name].queue_name,
      { label = "${name} — age of oldest message" }
    ]
  ]

  depth_metrics = [
    for name in local.arm_names : [
      "AWS/SQS", "ApproximateNumberOfMessagesVisible",
      "QueueName", module.arm[name].queue_name,
      { label = "${name} — queue depth" }
    ]
  ]

  capacity_metrics = [
    for name in local.arm_names : [
      "AWS/AutoScaling", "GroupInServiceInstances",
      "AutoScalingGroupName", module.arm[name].asg_name,
      { label = "${name} — in service" }
    ]
  ]

  arrival_metrics = [
    for name in local.arm_names : [
      "AWS/SQS", "NumberOfMessagesSent",
      "QueueName", module.arm[name].queue_name,
      { label = "${name} — arrivals" }
    ]
  ]
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = var.project

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Age of oldest message (the SLI)"
          metrics = local.age_metrics
          view    = "timeSeries"
          region  = var.region
          period  = 60
          stat    = "Maximum"
          yAxis   = { left = { label = "seconds", showUnits = false } }
          annotations = {
            horizontal = [{ label = "SLO", value = 60 }]
          }
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "In-service capacity"
          metrics = local.capacity_metrics
          view    = "timeSeries"
          region  = var.region
          period  = 60
          stat    = "Average"
          yAxis   = { left = { label = "instances", showUnits = false } }
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title   = "Arrivals per minute"
          metrics = local.arrival_metrics
          view    = "timeSeries"
          region  = var.region
          period  = 60
          stat    = "Sum"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title   = "Queue depth"
          metrics = local.depth_metrics
          view    = "timeSeries"
          region  = var.region
          period  = 60
          stat    = "Average"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 24
        height = 6
        properties = {
          title = "Forecast vs decision (custom arm)"
          metrics = [
            [local.metric_namespace, "ForecastArrivalsPerMinute",
            "AutoScalingGroupName", module.arm["custom"].asg_name, { label = "forecast" }],
            [local.metric_namespace, "CorrectedArrivalsPerMinute",
            "AutoScalingGroupName", module.arm["custom"].asg_name, { label = "forecast, level-corrected" }],
            [local.metric_namespace, "DesiredInstances",
              "AutoScalingGroupName", module.arm["custom"].asg_name,
            { label = "desired instances", yAxis = "right" }],
          ]
          view   = "timeSeries"
          region = var.region
          period = 60
          stat   = "Average"
          yAxis = {
            left  = { label = "arrivals/min", showUnits = false }
            right = { label = "instances", showUnits = false }
          }
        }
      },
    ]
  })
}
