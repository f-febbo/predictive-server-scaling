# Permissions, scoped to the specific queues and groups this stack creates.
# Nothing here uses a wildcard resource except CloudWatch reads and metric
# writes, neither of which supports resource-level scoping.

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# --- worker -----------------------------------------------------------------

resource "aws_iam_role" "worker" {
  name               = "${var.project}-worker"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

data "aws_iam_policy_document" "worker" {
  statement {
    sid = "ConsumeWork"
    actions = [
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
    ]
    resources = [for arm in module.arm : arm.queue_arn]
  }
}

resource "aws_iam_role_policy" "worker" {
  name   = "${var.project}-worker"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.worker.json
}

resource "aws_iam_instance_profile" "worker" {
  name = "${var.project}-worker"
  role = aws_iam_role.worker.name
}

# --- scaler Lambda ----------------------------------------------------------

resource "aws_iam_role" "scaler" {
  name               = "${var.project}-scaler"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

data "aws_iam_policy_document" "scaler" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:*"]
  }

  statement {
    sid = "ReadLoad"
    actions = [
      "cloudwatch:GetMetricStatistics",
      "cloudwatch:GetMetricData",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "PublishForecast"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
  }

  statement {
    sid       = "InspectQueue"
    actions   = ["sqs:GetQueueAttributes"]
    resources = [module.arm["custom"].queue_arn]
  }

  statement {
    sid = "SetCapacity"
    actions = [
      "autoscaling:SetDesiredCapacity",
      "autoscaling:DescribeAutoScalingGroups",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "scaler" {
  name   = "${var.project}-scaler"
  role   = aws_iam_role.scaler.id
  policy = data.aws_iam_policy_document.scaler.json
}

# --- load generator Lambda --------------------------------------------------

resource "aws_iam_role" "loadgen" {
  name               = "${var.project}-loadgen"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

data "aws_iam_policy_document" "loadgen" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:*"]
  }

  statement {
    sid       = "SendWork"
    actions   = ["sqs:SendMessage", "sqs:SendMessageBatch"]
    resources = [for arm in module.arm : arm.queue_arn]
  }
}

resource "aws_iam_role_policy" "loadgen" {
  name   = "${var.project}-loadgen"
  role   = aws_iam_role.loadgen.id
  policy = data.aws_iam_policy_document.loadgen.json
}

data "aws_caller_identity" "current" {}
