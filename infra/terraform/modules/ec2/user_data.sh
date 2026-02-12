#!/bin/bash
# EC2 User Data Script - VocalBaby Setup
set -euxo pipefail

# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
COMPOSE_VERSION="v2.23.0"
curl -L "https://github.com/docker/compose/releases/download/$${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Login to ECR
aws ecr get-login-password --region $(curl -s http://169.254.169.254/latest/meta-data/placement/region) \
  | docker login --username AWS --password-stdin ${ecr_repository_url}

# Pull and run latest image
docker pull ${ecr_repository_url}:latest
docker run -d \
  --name ${project_name} \
  --restart unless-stopped \
  -p 8000:8000 \
  ${ecr_repository_url}:latest

echo "VocalBaby deployment complete!"
