# Consumed by destroy.sh so teardown targets the right account resources.
output "region" {
  description = "Region everything was deployed to."
  value       = var.region
}

output "project" {
  description = "Resource name prefix."
  value       = var.project
}

output "queue_urls" {
  description = "Work queue per arm."
  value       = { for name, arm in module.arm : name => arm.queue_url }
}

output "asg_names" {
  description = "Auto Scaling group per arm."
  value       = { for name, arm in module.arm : name => arm.asg_name }
}

output "dashboard_url" {
  description = "The CloudWatch dashboard for the run."
  value       = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}

output "replay_start_utc" {
  description = "When the replay clock started. Minute 0 of the trace."
  value       = time_static.replay_start.rfc3339
}

output "replay_ends_utc" {
  description = "When the trace runs out and the queues go quiet."
  value       = timeadd(time_static.replay_start.rfc3339, "2880m")
}

output "estimated_hourly_cost_usd" {
  description = <<-EOT
    Rough upper bound while both arms run near peak, for orientation only.
    Assumes spot at about a third of on-demand. Verify against real pricing.
  EOT
  value       = format("%.2f", var.asg_max_size * length(module.arm) * 0.006)
}

output "teardown" {
  description = "How to stop paying for this."
  value       = "cd infra && ./destroy.sh"
}
