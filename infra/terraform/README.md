# VocalBaby Terraform Infrastructure

This directory contains Terraform modules for deploying VocalBaby on AWS.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │   ECR    │  │    S3    │  │       EC2 Instance        │  │
│  │ (Docker  │  │ (Models  │  │  ┌──────────────────────┐ │  │
│  │  Images) │  │  & Data) │  │  │   VocalBaby API      │ │  │
│  └──────────┘  └──────────┘  │  │   (Docker Container) │ │  │
│                              │  ├──────────────────────┤ │  │
│                              │  │   Prometheus          │ │  │
│                              │  ├──────────────────────┤ │  │
│                              │  │   Grafana             │ │  │
│                              │  └──────────────────────┘ │  │
│                              └──────────────────────────┘  │
│  ┌──────────────────────────┐                               │
│  │       IAM Roles          │                               │
│  │  (Least Privilege)       │                               │
│  └──────────────────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
infra/terraform/
├── modules/
│   ├── ecr/          # Container registry
│   ├── ec2/          # Compute instances
│   ├── s3/           # Storage buckets
│   ├── iam/          # IAM roles & policies
│   ├── networking/   # VPC, subnets, security groups
│   └── monitoring/   # Prometheus & Grafana
└── envs/
    └── production/   # Production environment config
```

## Usage

### Prerequisites

1. Install [Terraform](https://www.terraform.io/downloads) >= 1.5.0
2. Configure AWS credentials (via `aws configure` or environment variables)
3. An S3 bucket for Terraform state (optional but recommended)

### Deploy

```bash
cd infra/terraform/envs/production

# Initialize
terraform init

# Plan
terraform plan -var-file="terraform.tfvars"

# Apply
terraform apply -var-file="terraform.tfvars"
```

### Destroy

```bash
terraform destroy -var-file="terraform.tfvars"
```

## Security

- **No hardcoded credentials** — use AWS IAM roles, environment variables, or AWS Secrets Manager.
- All IAM policies follow least privilege principles.
- EC2 instances are in private subnets (when using full VPC setup).
- Security groups restrict access to specific ports.

## Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region | `us-east-1` |
| `project_name` | Project name for resource naming | `vocalbaby` |
| `environment` | Deployment environment | `production` |
| `instance_type` | EC2 instance type | `t3.medium` |
| `ecr_repo_name` | ECR repository name | `vocalbaby` |
| `s3_bucket_name` | S3 bucket for artifacts | `vocalbaby-artifacts` |
