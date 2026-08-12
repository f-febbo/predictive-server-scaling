# The shutdown backstop.
#
# Pins every fleet to zero some hours after the replay finishes, so that
# forgetting the run exists costs pennies rather than accumulating. This does
# not destroy anything -- `destroy.sh` is still required -- it only stops the
# compute meter.

resource "aws_iam_role" "watchdog" {
  name               = "${var.project}-watchdog"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

data "aws_iam_policy_document" "watchdog" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:*"]
  }

  statement {
    sid = "StopFleets"
    actions = [
      "autoscaling:UpdateAutoScalingGroup",
      "autoscaling:DescribeAutoScalingGroups",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "watchdog" {
  name   = "${var.project}-watchdog"
  role   = aws_iam_role.watchdog.id
  policy = data.aws_iam_policy_document.watchdog.json
}

resource "aws_lambda_function" "watchdog" {
  function_name = "${var.project}-watchdog"
  role          = aws_iam_role.watchdog.arn
  handler       = "watchdog_handler.handler"
  runtime       = "python3.12"
  timeout       = 30

  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256

  environment {
    variables = {
      REPLAY_START_EPOCH   = local.replay_start_epoch
      SHUTDOWN_AFTER_HOURS = tostring(var.shutdown_after_hours)
      ASG_NAMES            = join(",", [for arm in module.arm : arm.asg_name])
    }
  }
}

# Every five minutes rather than every minute: this only has to fire once, and
# a cheap check that runs 288 times a day is plenty.
resource "aws_cloudwatch_event_rule" "watchdog" {
  name                = "${var.project}-watchdog"
  description         = "Pins all fleets to zero once the experiment window closes."
  schedule_expression = "rate(5 minutes)"
}

resource "aws_cloudwatch_event_target" "watchdog" {
  rule      = aws_cloudwatch_event_rule.watchdog.name
  target_id = "watchdog"
  arn       = aws_lambda_function.watchdog.arn
}

resource "aws_lambda_permission" "watchdog" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.watchdog.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.watchdog.arn
}

resource "aws_cloudwatch_log_group" "watchdog" {
  name              = "/aws/lambda/${aws_lambda_function.watchdog.function_name}"
  retention_in_days = 3
}
