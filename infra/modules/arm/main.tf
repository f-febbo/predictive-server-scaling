# One complete scaling arm: a queue, a worker fleet, and nothing that decides
# how big the fleet should be.
#
# Both arms of the experiment are built from this module so that the queue, the
# worker, the instance type, and the load are identical between them. The only
# difference is what drives desired capacity — a custom forecasting Lambda in
# one arm, AWS native Predictive Scaling in the other. If anything else
# differed, the comparison would not be measuring what it claims to.

resource "aws_sqs_queue" "work" {
  name = "${var.name}-work"

  # Comfortably longer than service_seconds, so a merely slow worker does not
  # have its message handed to a second worker and processed twice.
  visibility_timeout_seconds = var.service_seconds * 6

  # Long enough that retention never truncates the measurement. This was
  # originally an hour, which silently destroyed the SLI: a backlogged queue
  # reported ApproximateAgeOfOldestMessage pinned at exactly 3600s while
  # messages older than that were deleted. A flat age metric read like a
  # stable system when it actually meant the opposite.
  message_retention_seconds = 14400

  # Long polling. Short polling bills an empty receive every few hundred
  # milliseconds per idle worker, which is the one way this stack could run up
  # an unexpected SQS bill.
  receive_wait_time_seconds = 20

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dead_letter.arn
    maxReceiveCount     = 3
  })
}

# Anything failing repeatedly lands here rather than cycling forever and
# inflating both the depth and the age metric the scaler reads.
resource "aws_sqs_queue" "dead_letter" {
  name                      = "${var.name}-dlq"
  message_retention_seconds = 86400
}

locals {
  # A worker: long-poll for one message, hold it for the service time, delete
  # it. One message at a time per instance, matching the simulator's model.
  #
  # Written against the AWS CLI that ships with Amazon Linux 2023, using
  # --query rather than jq so nothing needs installing at boot. Every package
  # install would add unpredictable seconds to the boot delay, which is the one
  # quantity this experiment is trying to measure.
  user_data = <<-EOT
    #!/bin/bash
    set -euo pipefail

    cat > /usr/local/bin/worker.sh <<'WORKER'
    #!/bin/bash
    QUEUE_URL="${aws_sqs_queue.work.url}"
    REGION="${data.aws_region.current.name}"

    # Application warmup: refuse work until the instance would realistically be
    # ready to serve it.
    sleep ${var.app_warmup_seconds}

    while true; do
      HANDLE=$(aws sqs receive-message \
        --queue-url "$QUEUE_URL" \
        --region "$REGION" \
        --wait-time-seconds 20 \
        --max-number-of-messages 1 \
        --query 'Messages[0].ReceiptHandle' \
        --output text 2>/dev/null || echo "None")

      if [ "$HANDLE" != "None" ] && [ -n "$HANDLE" ]; then
        sleep ${var.service_seconds}
        aws sqs delete-message \
          --queue-url "$QUEUE_URL" \
          --region "$REGION" \
          --receipt-handle "$HANDLE" >/dev/null 2>&1 || true
      fi
    done
    WORKER

    chmod +x /usr/local/bin/worker.sh

    cat > /etc/systemd/system/worker.service <<'UNIT'
    [Unit]
    Description=Queue worker
    After=network-online.target

    [Service]
    ExecStart=/usr/local/bin/worker.sh
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
    UNIT

    systemctl daemon-reload
    systemctl enable --now worker.service
  EOT
}

data "aws_region" "current" {}

resource "aws_launch_template" "worker" {
  name_prefix   = "${var.name}-"
  image_id      = var.ami_id
  instance_type = var.instance_type

  iam_instance_profile {
    name = var.instance_profile_name
  }

  vpc_security_group_ids = [var.security_group_id]
  user_data              = base64encode(local.user_data)

  # Spot purchasing is configured on the ASG's mixed_instances_policy rather
  # than here, so that the group can fall back across instance types. Setting
  # instance_market_options as well would conflict with it.

  monitoring {
    # EC2 detailed monitoring is deliberately OFF. It bills about $2.10 per
    # instance-month, which on a fleet this churny would be a large share of
    # the total, and nothing here reads per-instance metrics: the scaler works
    # from SQS metrics and ASG group metrics, both of which are already
    # one-minute and free.
    enabled = false
  }

  tag_specifications {
    resource_type = "instance"
    tags          = { Name = "${var.name}-worker" }
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "workers" {
  name                = "${var.name}-asg"
  vpc_zone_identifier = var.subnet_ids

  min_size         = var.min_size
  max_size         = var.max_size
  desired_capacity = var.desired_capacity

  # Instances are interchangeable and hold no state, so there is no reason to
  # wait on health checks beyond the boot itself.
  health_check_type         = "EC2"
  health_check_grace_period = 120

  # Group-level metrics at one-minute granularity, which the dashboard and the
  # native predictive scaling policy both read. Free, unlike the per-instance
  # detailed monitoring deliberately left off in the launch template.
  metrics_granularity = "1Minute"
  enabled_metrics = [
    "GroupInServiceInstances",
    "GroupPendingInstances",
    "GroupTerminatingInstances",
    "GroupDesiredCapacity",
    "GroupTotalInstances",
  ]

  # A single spot instance type in a single family is a single point of
  # failure. Mid-run this group spent hours unable to launch anything --
  # "We currently do not have sufficient t4g.small capacity in the
  # Availability Zone you requested" -- while its queue backed up. That is a
  # capacity result, not a scaling result, and it contaminates the comparison.
  #
  # BUT: widening the list is not free on every account. The first attempt at
  # this fix added six larger Graviton types, and every launch of them failed
  # with "The specified instance type is not eligible for Free Tier" -- an AWS
  # Free Tier account may only launch free-tier-eligible types, so the fix made
  # the outage worse rather than better. See var.instance_types.
  mixed_instances_policy {
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.worker.id
        version            = "$Latest"
      }

      dynamic "override" {
        for_each = var.instance_types
        content {
          instance_type = override.value
        }
      }
    }

    instances_distribution {
      # One on-demand instance per arm, so the fleet can never be driven to
      # zero by a spot shortage. Everything above that is spot.
      on_demand_base_capacity                  = 1
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "capacity-optimized"
    }
  }

  # Terraform must not fight the scaler over desired capacity, so that field is
  # owned by whatever drives this arm once the group exists.
  #
  # min_size and max_size stay under Terraform's control, because they have to
  # be tunable against the account's spot quota. The cost is that running
  # `terraform apply` AFTER the shutdown watchdog has pinned the group to zero
  # would restore the bounds and relaunch instances. Run `destroy.sh` at the
  # end of an experiment, not `apply`.
  lifecycle {
    ignore_changes = [desired_capacity]
  }

  tag {
    key                 = "Name"
    value               = "${var.name}-worker"
    propagate_at_launch = true
  }

  # Tagged explicitly because the provider's default_tags do NOT reach
  # instances launched by an Auto Scaling group -- the ASG service launches
  # them, not Terraform. Without this, instances carry no Project tag, and any
  # teardown check that filters on one reports a clean account while the fleet
  # is still running.
  tag {
    key                 = "Project"
    value               = var.project
    propagate_at_launch = true
  }

  timeouts {
    delete = "15m"
  }
}
