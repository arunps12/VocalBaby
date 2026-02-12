# ---------------------------------------------------------
# VocalBaby — S3 Module
# Storage for models, artifacts, and data
# ---------------------------------------------------------

variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "bucket_name" {
  type = string
}

# ---------------------------------------------------------
# S3 Bucket
# ---------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket        = var.bucket_name
  force_destroy = false

  tags = {
    Name        = "${var.project_name}-artifacts"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ---------------------------------------------------------
# Versioning
# ---------------------------------------------------------
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ---------------------------------------------------------
# Encryption
# ---------------------------------------------------------
resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------------------------------------------------------
# Block Public Access
# ---------------------------------------------------------
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------
# Outputs
# ---------------------------------------------------------
output "bucket_name" {
  value = aws_s3_bucket.artifacts.id
}

output "bucket_arn" {
  value = aws_s3_bucket.artifacts.arn
}
