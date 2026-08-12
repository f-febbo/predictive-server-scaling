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
  message_retention_seconds  = 3600

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

  # Spot, but without a max price: the default is the on-demand price, and
  # naming a lower ceiling is the usual way to get silently starved of capacity
  # and mistake it for a scaling bug.
  instance_market_options {
    market_type = "spot"

    spot_options {
      spot_instance_type             = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }

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

  launch_template {
    id      = aws_launch_template.worker.id
    version = "$Latest"
  }

  # Terraform must not fight the scaler over desired capacity; whatever is
  # driving this arm owns it after creation. min_size and max_size are ignored
  # too because the shutdown watchdog pins them to zero when the experiment
  # window closes, and a later plan should not offer to undo that.
  lifecycle {
    ignore_changes = [desired_capacity, min_size, max_size]
  }

  tag {
    key                 = "Name"
    value               = "${var.name}-worker"
    propagate_at_launch = true
  }

  timeouts {
    delete = "15m"
  }
}
