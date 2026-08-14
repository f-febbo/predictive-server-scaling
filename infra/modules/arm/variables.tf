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
