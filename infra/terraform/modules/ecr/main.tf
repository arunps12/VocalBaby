# ---------------------------------------------------------
# VocalBaby — ECR Module
# Amazon Elastic Container Registry
# ---------------------------------------------------------

variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "image_tag_mutability" {
  type    = string
  default = "MUTABLE"
}

# ---------------------------------------------------------
# ECR Repository
# ---------------------------------------------------------
resource "aws_ecr_repository" "main" {
  name                 = var.project_name
  image_tag_mutability = var.image_tag_mutability
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = "${var.project_name}-ecr"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ---------------------------------------------------------
# Lifecycle Policy - Keep last 10 images
# ---------------------------------------------------------
resource "aws_ecr_lifecycle_policy" "main" {
  repository = aws_ecr_repository.main.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ---------------------------------------------------------
# Outputs
# ---------------------------------------------------------
output "repository_url" {
  value = aws_ecr_repository.main.repository_url
}

output "repository_name" {
  value = aws_ecr_repository.main.name
}

output "registry_id" {
  value = aws_ecr_repository.main.registry_id
}
