# Networking, deliberately minimal.
#
# There is no NAT Gateway anywhere in this stack. A NAT Gateway costs roughly
# $32/month whether or not anything flows through it, which would dwarf every
# other line item here and is the single most common way a demo stack quietly
# runs up a bill. Workers sit in a public subnet with public IPs and reach SQS
# over the internet gateway instead. For a fleet of sleep-loop workers holding
# no data, that trade is fine; a production system would use private subnets
# with VPC endpoints.

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = var.project }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = { Name = var.project }
}

# Two subnets in different AZs: spot capacity is much easier to get when the
# ASG can choose between zones, and a capacity shortfall in one AZ would
# otherwise look like a scaling failure.
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${var.project}-public-${count.index}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${var.project}-public" }
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "worker" {
  name        = "${var.project}-worker"
  description = "Worker egress to SQS and CloudWatch; no inbound by default."
  vpc_id      = aws_vpc.main.id

  egress {
    description = "Outbound to AWS APIs."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project}-worker" }
}

# Optional and off by default. The workers are driven entirely by the queue and
# report through CloudWatch, so there is nothing to log in for.
resource "aws_vpc_security_group_ingress_rule" "ssh" {
  count = var.allowed_ssh_cidr == "" ? 0 : 1

  security_group_id = aws_security_group.worker.id
  description       = "SSH for debugging."
  cidr_ipv4         = var.allowed_ssh_cidr
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
}
