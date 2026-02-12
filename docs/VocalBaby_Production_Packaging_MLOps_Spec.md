# VocalBaby -- Production Architecture & Packaging Specification

Generated on: 2026-02-12 11:45:32 UTC

------------------------------------------------------------------------

# OBJECTIVE

Refactor the VocalBaby repository into a production-grade ML system
that:

1.  Uses src-layout packaging
2.  Renames visioninfantnet → vocalbaby
3.  Ships as a reusable Python library
4.  Provides a runnable server via: vocalbaby-serve
5.  Integrates DVC stage-wise pipeline
6.  Adds Prometheus monitoring
7.  Adds Grafana dashboards
8.  Adds Evidently drift detection
9.  Uses Terraform for Infrastructure as Code
10. Maintains AWS deployment compatibility

This document is the SINGLE SOURCE OF TRUTH for the code agent.

------------------------------------------------------------------------

# PHASE 0 --- STRUCTURE REFACTOR (MANDATORY)

Rename: visioninfantnet → vocalbaby

Move package under src layout:

Target structure:

VocalBaby/ │ ├── src/ │ └── vocalbaby/ │ ├── **init**.py │ ├──
components/ │ ├── pipeline/ │ ├── utils/ │ ├── entity/ │ ├── exception/
│ ├── logging/ │ ├── cloud/ │ ├── monitoring/ │ │ ├── drift.py │ │ └──
metrics.py │ ├── api/ │ │ ├── app.py │ │ └── routes.py │ └── cli.py │
├── configs/ │ ├── model.yaml │ ├── training.yaml │ ├── drift.yaml │ └──
monitoring.yaml │ ├── scripts/ │ ├── run_ingestion.sh │ ├──
run_validation.sh │ ├── run_training.sh │ ├── run_inference.sh │ ├──
run_drift.sh │ ├── infra/ │ └── terraform/ │ ├── modules/ │ └── envs/ │
├── artifacts/ ├── tests/ ├── dvc.yaml ├── Dockerfile ├──
docker-compose.monitoring.yml ├── pyproject.toml └── README.md

------------------------------------------------------------------------

# PHASE 1 --- PACKAGING (Library + Runnable Server)

Use src-layout packaging.

In pyproject.toml:

\[project.scripts\] vocalbaby-serve = "vocalbaby.cli:serve"

Create src/vocalbaby/cli.py:

def serve(): import uvicorn uvicorn.run( "vocalbaby.api.app:app",
host="0.0.0.0", port=8000, reload=False, )

Users should be able to:

pip install vocalbaby vocalbaby-serve

------------------------------------------------------------------------

# PHASE 2 --- DVC PIPELINE

Create stage-wise dvc.yaml:

stages: ingestion: cmd: bash scripts/run_ingestion.sh validation: cmd:
bash scripts/run_validation.sh training: cmd: bash
scripts/run_training.sh drift: cmd: bash scripts/run_drift.sh

All artifacts stored under artifacts/ and tracked.

------------------------------------------------------------------------

# PHASE 3 --- PROMETHEUS MONITORING

Add Prometheus metrics in FastAPI:

-   Request counter
-   Latency histogram
-   Error counter
-   Model version label

Expose endpoint:

/metrics

Integrate prometheus_client.

------------------------------------------------------------------------

# PHASE 4 --- GRAFANA

Provide dashboard JSON definitions for:

-   Request rate
-   Latency
-   Error rate
-   Drift score
-   Model version usage

------------------------------------------------------------------------

# PHASE 5 --- EVIDENTLY DRIFT MONITORING

Implement:

src/vocalbaby/monitoring/drift.py

Features:

-   Compare reference vs production batch
-   Generate HTML drift report
-   Store in artifacts/drift/
-   Export drift score to Prometheus

Schedule via:

-   GitHub Actions
-   Cron job
-   Kubernetes CronJob (if applicable)

------------------------------------------------------------------------

# PHASE 6 --- TERRAFORM (Infrastructure as Code)

Provision via Terraform:

Networking: - VPC - Subnets - Security groups

Compute: - EC2 instances - Optional Auto Scaling

Container: - Amazon ECR

Storage: - S3 buckets for models and artifacts

Monitoring: - Prometheus instance - Grafana instance

IAM: - Least privilege roles

All infra modularized.

------------------------------------------------------------------------

# PHASE 7 --- CI/CD EXTENSION

GitHub Actions must:

-   Run lint + tests
-   Run DVC repro
-   Build Docker image
-   Push to ECR
-   Deploy to EC2
-   Run scheduled drift checks

------------------------------------------------------------------------

# FINAL STATE

After implementation, VocalBaby becomes:

-   A pip-installable library
-   A runnable FastAPI server via vocalbaby-serve
-   Fully DVC versioned
-   Prometheus monitored
-   Grafana visualized
-   Evidently drift aware
-   Terraform managed infrastructure
-   AWS deployable production system

End of specification.
