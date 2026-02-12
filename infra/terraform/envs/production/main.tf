# ---------------------------------------------------------
# VocalBaby — Production Environment
# Root Terraform Configuration
# ---------------------------------------------------------

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use S3 backend for state management
  # backend "s3" {
  #   bucket         = "vocalbaby-terraform-state"
  #   key            = "production/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "vocalbaby-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ---------------------------------------------------------
# Variables
# ---------------------------------------------------------
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "vocalbaby"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "s3_bucket_name" {
  description = "S3 bucket for artifacts and models"
  type        = string
  default     = "vocalbaby-artifacts"
}

variable "key_name" {
  description = "EC2 key pair name for SSH access (optional)"
  type        = string
  default     = ""
}

# ---------------------------------------------------------
# Modules
# ---------------------------------------------------------

module "networking" {
  source       = "../../modules/networking"
  project_name = var.project_name
  environment  = var.environment
}

module "ecr" {
  source       = "../../modules/ecr"
  project_name = var.project_name
  environment  = var.environment
}

module "s3" {
  source       = "../../modules/s3"
  project_name = var.project_name
  environment  = var.environment
  bucket_name  = var.s3_bucket_name
}

module "iam" {
  source        = "../../modules/iam"
  project_name  = var.project_name
  environment   = var.environment
  s3_bucket_arn = module.s3.bucket_arn
}

module "ec2" {
  source               = "../../modules/ec2"
  project_name         = var.project_name
  environment          = var.environment
  instance_type        = var.instance_type
  subnet_id            = module.networking.public_subnet_ids[0]
  security_group_ids   = [module.networking.app_security_group_id]
  iam_instance_profile = module.iam.instance_profile_name
  key_name             = var.key_name
  ecr_repository_url   = module.ecr.repository_url
}

module "monitoring" {
  source       = "../../modules/monitoring"
  project_name = var.project_name
  environment  = var.environment
  app_target   = "${module.ec2.private_ip}:8000"
}

# ---------------------------------------------------------
# Outputs
# ---------------------------------------------------------
output "ecr_repository_url" {
  description = "ECR repository URL for Docker images"
  value       = module.ecr.repository_url
}

output "ec2_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = module.ec2.public_ip
}

output "s3_bucket_name" {
  description = "S3 bucket for artifacts"
  value       = module.s3.bucket_name
}

output "api_url" {
  description = "VocalBaby API endpoint"
  value       = "http://${module.ec2.public_ip}:8000"
}

output "grafana_url" {
  description = "Grafana dashboard URL"
  value       = "http://${module.ec2.public_ip}:3000"
}

output "prometheus_url" {
  description = "Prometheus URL"
  value       = "http://${module.ec2.public_ip}:9090"
}
