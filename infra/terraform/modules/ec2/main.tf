# ---------------------------------------------------------
# VocalBaby — EC2 Module
# Compute instances for running VocalBaby
# ---------------------------------------------------------

variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "instance_type" {
  type    = string
  default = "t3.medium"
}

variable "subnet_id" {
  type = string
}

variable "security_group_ids" {
  type = list(string)
}

variable "iam_instance_profile" {
  type = string
}

variable "key_name" {
  type    = string
  default = ""
}

variable "ecr_repository_url" {
  type = string
}

# ---------------------------------------------------------
# Get latest Amazon Linux 2 AMI
# ---------------------------------------------------------
data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ---------------------------------------------------------
# EC2 Instance
# ---------------------------------------------------------
resource "aws_instance" "app" {
  ami                    = data.aws_ami.amazon_linux_2.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = var.security_group_ids
  iam_instance_profile   = var.iam_instance_profile

  key_name = var.key_name != "" ? var.key_name : null

  root_block_device {
    volume_type = "gp3"
    volume_size = 30
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    ecr_repository_url = var.ecr_repository_url
    project_name       = var.project_name
  })

  tags = {
    Name        = "${var.project_name}-app"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ---------------------------------------------------------
# Outputs
# ---------------------------------------------------------
output "instance_id" {
  value = aws_instance.app.id
}

output "public_ip" {
  value = aws_instance.app.public_ip
}

output "private_ip" {
  value = aws_instance.app.private_ip
}
