# The comparison arm: AWS native Predictive Scaling on a custom SQS metric.
#
# Using the managed feature and measuring where it falls short is a much
# stronger position than implying AWS has no such capability. It does, it works,
# and for daily and weekly cycles it is a reasonable choice that costs nothing
# to operate.
#
# The gap this project targets is granularity. Native Predictive Scaling
# forecasts on an HOURLY grid: it produces one capacity figure per hour, from at
# least 24 hours of history and ideally 14 days. That is well matched to a
# smooth daily cycle and structurally unable to address a burst that arrives and
# clears inside an hour. The custom arm forecasts at one-minute resolution over
# a 15-minute horizon, which is the timescale instance boot delay actually
# operates on.
#
# KNOWN LIMITATION OF A SHORT RUN: with less than 24 hours of metric history
# this policy emits no forecast at all, and for the first several days its
# forecast is built on very little data. A 48-hour run therefore demonstrates
# that the integration is correct, not that the managed feature is at its best.
# Any comparison drawn from a short run must say so.
#
# NOTE ON METRIC MATH: every metric below is addressed directly through
# metric_stat rather than through a SEARCH() expression. Target tracking
# policies are backed by CloudWatch metric alarms, and alarms reject SEARCH
# ("SEARCH is not supported on Metric Alarms"). SEARCH exists for wildcard
# discovery, which is unnecessary here because the queue and group names are
# known when the plan is generated.

locals {
  # Backlog per instance that keeps the oldest message inside the latency
  # budget, from Little's Law: a 60s budget divided by the service time is how
  # many messages a worker can clear in time.
  backlog_target = var.service_seconds > 0 ? 60 / var.service_seconds : 2
}

resource "aws_autoscaling_policy" "native_predictive" {
  count = var.enable_native_predictive_scaling ? 1 : 0

  name                   = "${var.project}-native-predictive"
  autoscaling_group_name = module.arm["native"].asg_name
  policy_type            = "PredictiveScaling"

  predictive_scaling_configuration {
    # Forecast and act on it, buffering capacity slightly ahead of the
    # predicted need so instances finish booting before the load lands.
    mode                         = "ForecastAndScale"
    scheduling_buffer_time       = 300
    max_capacity_breach_behavior = "IncreaseMaxCapacity"
    max_capacity_buffer          = 10

    metric_specification {
      target_value = local.backlog_target

      # Total load offered to the fleet.
      customized_load_metric_specification {
        metric_data_queries {
          id = "load"

          metric_stat {
            metric {
              namespace   = "AWS/SQS"
              metric_name = "NumberOfMessagesSent"

              dimensions {
                name  = "QueueName"
                value = module.arm["native"].queue_name
              }
            }

            stat = "Sum"
          }
        }
      }

      # How much capacity was actually in service.
      customized_capacity_metric_specification {
        metric_data_queries {
          id = "capacity"

          metric_stat {
            metric {
              namespace   = "AWS/AutoScaling"
              metric_name = "GroupInServiceInstances"

              dimensions {
                name  = "AutoScalingGroupName"
                value = module.arm["native"].asg_name
              }
            }

            stat = "Average"
          }
        }
      }

      # The utilisation signal: backlog divided by in-service capacity.
      customized_scaling_metric_specification {
        metric_data_queries {
          id          = "backlog"
          return_data = false

          metric_stat {
            metric {
              namespace   = "AWS/SQS"
              metric_name = "ApproximateNumberOfMessagesVisible"

              dimensions {
                name  = "QueueName"
                value = module.arm["native"].queue_name
              }
            }

            stat = "Average"
          }
        }

        metric_data_queries {
          id          = "instances"
          return_data = false

          metric_stat {
            metric {
              namespace   = "AWS/AutoScaling"
              metric_name = "GroupInServiceInstances"

              dimensions {
                name  = "AutoScalingGroupName"
                value = module.arm["native"].asg_name
              }
            }

            stat = "Average"
          }
        }

        metric_data_queries {
          id    = "backlog_per_instance"
          label = "Backlog per instance"
          # Plain division, as AWS documents for this pattern. If the group is
          # ever at zero instances the expression yields no data point rather
          # than an error, which the alarm treats as missing data.
          expression = "backlog / instances"
        }
      }
    }
  }
}

# Predictive scaling handles the anticipated shape; target tracking handles
# whatever the hourly forecast missed. AWS recommends pairing them, and without
# it this arm would have no way to respond to a burst inside the hour -- which
# would make the comparison a strawman rather than a measurement.
resource "aws_autoscaling_policy" "native_target_tracking" {
  count = var.enable_native_predictive_scaling ? 1 : 0

  name                   = "${var.project}-native-target-tracking"
  autoscaling_group_name = module.arm["native"].asg_name
  policy_type            = "TargetTrackingScaling"

  target_tracking_configuration {
    target_value = local.backlog_target

    customized_metric_specification {
      metrics {
        id          = "backlog"
        return_data = false

        metric_stat {
          metric {
            namespace   = "AWS/SQS"
            metric_name = "ApproximateNumberOfMessagesVisible"

            dimensions {
              name  = "QueueName"
              value = module.arm["native"].queue_name
            }
          }

          stat = "Average"
        }
      }

      metrics {
        id          = "instances"
        return_data = false

        metric_stat {
          metric {
            namespace   = "AWS/AutoScaling"
            metric_name = "GroupInServiceInstances"

            dimensions {
              name  = "AutoScalingGroupName"
              value = module.arm["native"].asg_name
            }
          }

          stat = "Average"
        }
      }

      metrics {
        id          = "backlog_per_instance"
        expression  = "backlog / instances"
        return_data = true
      }
    }
  }
}
