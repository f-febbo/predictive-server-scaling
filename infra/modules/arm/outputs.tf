output "queue_url" {
  value = aws_sqs_queue.work.url
}

output "queue_arn" {
  value = aws_sqs_queue.work.arn
}

output "queue_name" {
  value = aws_sqs_queue.work.name
}

output "asg_name" {
  value = aws_autoscaling_group.workers.name
}

output "asg_arn" {
  value = aws_autoscaling_group.workers.arn
}
