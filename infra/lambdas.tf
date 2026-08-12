# The two Lambdas: one generates load, one scales the custom arm.
#
# Both tick once a minute, matching the simulator's scaler interval and the
# one-minute resolution of the trace.

# Pinned at first apply so the replay has a fixed origin. Using timestamp()
# directly would move the origin on every plan and silently restart the replay.
resource "time_static" "replay_start" {}

data "archive_file" "lambda" {
  type        = "zip"
  source_dir  = "${path.module}/lambda"
  output_path = "${path.module}/.build/lambda.zip"
}

locals {
  replay_start_epoch = tostring(time_static.replay_start.unix)
  metric_namespace   = "PredictiveAutoscaler"
}

# --- load generator ---------------------------------------------------------

resource "aws_lambda_function" "loadgen" {
  function_name = "${var.project}-loadgen"
  role          = aws_iam_role.loadgen.arn
  handler       = "loadgen_handler.handler"
  runtime       = "python3.12"
  timeout       = 50 # under the one-minute tick, so invocations cannot overlap

  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256

  environment {
    variables = {
      REPLAY_START_EPOCH = local.replay_start_epoch
      ARRIVAL_DIVISOR    = tostring(var.arrival_divisor)
      QUEUE_URLS         = join(",", [for arm in module.arm : arm.queue_url])
    }
  }
}

# --- predictive scaler ------------------------------------------------------

resource "aws_lambda_function" "scaler" {
  function_name = "${var.project}-scaler"
  role          = aws_iam_role.scaler.arn
  handler       = "scaler_handler.handler"
  runtime       = "python3.12"
  timeout       = 30

  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256

  environment {
    variables = {
      ASG_NAME           = module.arm["custom"].asg_name
      QUEUE_NAME         = module.arm["custom"].queue_name
      METRIC_NAMESPACE   = local.metric_namespace
      REPLAY_START_EPOCH = local.replay_start_epoch
      SERVICE_SECONDS    = tostring(var.service_seconds)
      TARGET_UTILIZATION = tostring(var.target_utilization)
      ARRIVAL_DIVISOR    = tostring(var.arrival_divisor)
      MIN_SIZE           = tostring(var.asg_min_size)
      MAX_SIZE           = tostring(var.asg_max_size)
    }
  }
}

# --- schedules --------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "every_minute" {
  name                = "${var.project}-every-minute"
  description         = "Drives the load generator and the scaler."
  schedule_expression = "rate(1 minute)"
}

resource "aws_cloudwatch_event_target" "loadgen" {
  rule      = aws_cloudwatch_event_rule.every_minute.name
  target_id = "loadgen"
  arn       = aws_lambda_function.loadgen.arn
}

resource "aws_cloudwatch_event_target" "scaler" {
  rule      = aws_cloudwatch_event_rule.every_minute.name
  target_id = "scaler"
  arn       = aws_lambda_function.scaler.arn
}

resource "aws_lambda_permission" "loadgen" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.loadgen.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.every_minute.arn
}

resource "aws_lambda_permission" "scaler" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scaler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.every_minute.arn
}

# Short retention: these logs are for watching a two-day experiment, and
# ingestion is billed per GB.
resource "aws_cloudwatch_log_group" "loadgen" {
  name              = "/aws/lambda/${aws_lambda_function.loadgen.function_name}"
  retention_in_days = 3
}

resource "aws_cloudwatch_log_group" "scaler" {
  name              = "/aws/lambda/${aws_lambda_function.scaler.function_name}"
  retention_in_days = 3
}
