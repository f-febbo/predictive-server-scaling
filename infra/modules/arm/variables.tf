variable "project" {
  description = "Project tag value."
  type        = string
}

variable "name" {
  description = "Full resource name prefix for this arm."
  type        = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

variable "security_group_id" {
  type = string
}

variable "instance_profile_name" {
  type = string
}

variable "instance_type" {
  type = string
}

variable "ami_id" {
  type = string
}

variable "service_seconds" {
  description = "Seconds a worker spends on each message."
  type        = number
}

variable "app_warmup_seconds" {
  description = <<-EOT
    Artificial delay before a freshly booted worker accepts its first message.

    This models application warmup — JIT, cache fill, connection pools — which
    is a real part of the boot delay the whole project is about. EC2 launch
    alone is roughly 60-90s; adding this brings the observed end-to-end boot
    delay near the 180s the simulator assumed, so the live run and the offline
    study describe the same system.
  EOT
  type        = number
}

variable "min_size" {
  type = number
}

variable "max_size" {
  type = number
}

variable "desired_capacity" {
  type = number
}

variable "instance_types" {
  description = <<-EOT
    Instance types the group may launch, best-first.

    Widening this improves spot fill rates, but two constraints bind together:

      1. The AMI is arm64, so only Graviton types are usable.
      2. An AWS Free Tier account may launch ONLY free-tier-eligible types.
         Anything else fails with "The specified instance type is not eligible
         for Free Tier" -- every launch, immediately, no matter the capacity.

    On a Free Tier account those two rules intersect to leave exactly
    t4g.small and t4g.micro. Check before widening:

        aws ec2 describe-instance-types --region us-east-1           --filters "Name=free-tier-eligible,Values=true"           --query "InstanceTypes[].InstanceType"

    On a normal account, adding t4g.medium, m6g.medium, and c6g.medium gives
    far better spot availability.
  EOT
  type        = list(string)
  default     = ["t4g.small", "t4g.micro"]
}
