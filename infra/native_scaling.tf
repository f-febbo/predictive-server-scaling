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

resource "aws_autoscaling_policy" "native_predictive" {
  count = var.enable_native_predictive_scaling ? 1 : 0

  name                   = "${var.project}-native-predictive"
  autoscaling_group_name = module.arm["native"].asg_name
  policy_type            = "PredictiveScaling"

  predictive_scaling_configuration {
    # Scale out ahead of the forecast but never scale in on it; scaling in is
    # left to the target-tracking policy below, which sees the real queue.
    mode                         = "ForecastAndScale"
    scheduling_buffer_time       = 300
    max_capacity_breach_behavior = "IncreaseMaxCapacity"
    max_capacity_buffer          = 10

    metric_specification {
      # Backlog per instance that keeps the oldest message inside the latency
      # budget: 60s budget / 30s per message = 2 messages per worker.
      target_value = var.service_seconds > 0 ? 60 / var.service_seconds : 2

      # Total load offered to the fleet.
      customized_load_metric_specification {
        metric_data_queries {
          id          = "load"
          expression  = "SUM(SEARCH('{AWS/SQS,QueueName} MetricName=\"NumberOfMessagesSent\" QueueName=\"${module.arm["native"].queue_name}\"', 'Sum'))"
          label       = "Messages enqueued"
          return_data = true
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
          expression  = "SUM(SEARCH('{AWS/SQS,QueueName} MetricName=\"ApproximateNumberOfMessagesVisible\" QueueName=\"${module.arm["native"].queue_name}\"', 'Average'))"
          label       = "Messages visible"
          return_data = false
        }

        metric_data_queries {
          id = "instances"

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

          return_data = false
        }

        metric_data_queries {
          id         = "backlog_per_instance"
          expression = "backlog / MAX([instances, 1])"
          label      = "Backlog per instance"
        }
      }
    }
  }
}

# Predictive scaling handles the anticipated shape; target tracking handles
# whatever the hourly forecast missed. AWS recommends pairing them, and without
# it this arm would have no way to respond to a burst inside the hour — which
# would make the comparison a strawman rather than a measurement.
resource "aws_autoscaling_policy" "native_target_tracking" {
  count = var.enable_native_predictive_scaling ? 1 : 0

  name                   = "${var.project}-native-target-tracking"
  autoscaling_group_name = module.arm["native"].asg_name
  policy_type            = "TargetTrackingScaling"

  target_tracking_configuration {
    target_value = var.service_seconds > 0 ? 60 / var.service_seconds : 2

    customized_metric_specification {
      metrics {
        id          = "backlog"
        expression  = "SUM(SEARCH('{AWS/SQS,QueueName} MetricName=\"ApproximateNumberOfMessagesVisible\" QueueName=\"${module.arm["native"].queue_name}\"', 'Average'))"
        return_data = false
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
        expression  = "backlog / MAX([instances, 1])"
        return_data = true
      }
    }
  }
}
