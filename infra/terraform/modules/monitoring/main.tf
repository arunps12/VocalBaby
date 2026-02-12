# ---------------------------------------------------------
# VocalBaby — Monitoring Module
# Prometheus & Grafana on EC2 (via Docker Compose)
# ---------------------------------------------------------

variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "app_target" {
  description = "Target host:port for the VocalBaby API (for Prometheus scraping)"
  type        = string
}

# This module outputs the Prometheus and Grafana configuration
# that should be deployed alongside the application.
# In a production setup, these would be separate EC2 instances
# or managed services (Amazon Managed Prometheus/Grafana).

output "prometheus_config" {
  value = {
    scrape_interval = "15s"
    targets         = [var.app_target]
    job_name        = "vocalbaby-api"
    metrics_path    = "/metrics"
  }
}

output "grafana_config" {
  value = {
    admin_user     = "admin"
    dashboard_path = "/var/lib/grafana/dashboards"
    port           = 3000
  }
}
