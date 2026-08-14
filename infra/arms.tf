# The two arms of the experiment.
#
# Identical queues, workers, instance types, and load. The only difference is
# what sets desired capacity: a forecasting Lambda in the "custom" arm, AWS
# native Predictive Scaling in the "native" arm. Running them side by side on
# separate queues — rather than pointing both at one queue — is what makes the
# comparison meaningful, since two scalers draining the same queue would each
# be reacting to the other's work.

# Amazon Linux 2023 for the instance architecture implied by the instance type.
data "aws_ssm_parameter" "al2023" {
  name = startswith(var.instance_type, "t4g.") || startswith(var.instance_type, "m7g.") ? "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64" : "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

locals {
  arms = var.enable_native_predictive_scaling ? ["custom", "native"] : ["custom"]
}

module "arm" {
  source   = "./modules/arm"
  for_each = toset(local.arms)

  name                  = "${var.project}-${each.key}"
  project               = var.project
  vpc_id                = aws_vpc.main.id
  subnet_ids            = aws_subnet.public[*].id
  security_group_id     = aws_security_group.worker.id
  instance_profile_name = aws_iam_instance_profile.worker.name
  instance_type         = var.instance_type
  ami_id                = data.aws_ssm_parameter.al2023.value

  service_seconds    = var.service_seconds
  app_warmup_seconds = var.app_warmup_seconds

  min_size         = var.asg_min_size
  max_size         = var.asg_max_size
  desired_capacity = var.asg_min_size
}
