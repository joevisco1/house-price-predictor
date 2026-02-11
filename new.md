🏠 House Price Predictor

**An End-to-End MLOps System Using Kubernetes, Argo, and KEDA**

This repository contains a fully operational MLOps system built end-to-end — from raw data to production-grade model deployment, observability, controlled rollouts, autoscaling, and drift detection.

I started this project about two months ago and built it piece by piece, often unsure if I’d make it through. It turned into a monster — but the right kind of monster. The kind that solidifies MLOps in a way tutorials never do.

I’ll probably rebuild this system again (and again) until I can do it from memory.

This project began from the excellent Udemy course “Ultimate DevOps to MLOps Bootcamp” by Gourav J. Shah, and then diverged significantly. I extended it, broke it, fixed it, and eventually got it running like a real production system.

![](media/0db89a0b76d7d8db70f5afad79e44edd.png)

## 📦 Project Structure

```
house-price-predictor/
├── configs/                # YAML-based configuration for models
├── data/                   # Raw and processed datasets
├── deployment/
│   └── mlflow/             # Docker Compose setup for MLflow
├── models/                 # Trained models and preprocessors
├── notebooks/              # Optional Jupyter notebooks for experimentation
├── src/
│   ├── data/               # Data cleaning and preprocessing scripts
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Model training and evaluation
├── requirements.txt        # Python dependencies
└── README.md               # You’re here
```

🛠️ Local Development & Learning Environment

## Prerequisites

-   Python 3.11
-   Git
-   Visual Studio Code (or your preferred editor)
-   uv (Python package & environment manager)
-   Docker Desktop (or Podman Desktop)

## Environment Setup

1️⃣ Fork & Clone

```
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
```

2️⃣ Create Virtual Environment (UV)

```
uv venv --python python3.11
source .venv/bin/activate
```

3️⃣ Install Dependencies

```
uv pip install -r requirements.txt
```

📊 \# MLflow Experiment Tracking

MLflow is used for experiment tracking, metrics, artifacts, and model versioning.

```
cd deployment/mlflow
docker compose -f mlflow-docker-compose.yml up -d
docker compose ps
```

Using Podman?

```
podman compose -f mlflow-docker-compose.yml up -d
podman compose ps
```

Access MLflow UI:

```
👉 http://localhost:5555
```

📒 Optional: JupyterLab

```
uv python -m jupyterlab
# or
python -m jupyterlab
```

## Technologies Involved

-   **Python / FastAPI** — inference API (single & batch predictions)
-   **Streamlit** — UI layer (client of the inference API)
-   **MLflow** — experiments, artifacts, metrics, model registry
-   **Docker** — reproducible builds and runtime isolation
-   **Kubernetes** — orchestration (Deployments, Services, ConfigMaps)
-   **GitHub Actions** — CI pipelines
-   **Argo CD** — GitOps deployment & drift detection
-   **Argo Rollouts** — blue/green & canary deployments
-   **Prometheus** — metrics scraping and storage
-   **Grafana** — visualization
-   **KEDA + HPA** — autoscaling based on real workload signals
-   **Observability patterns** — metrics, health checks, feedback loops

## From First to Last: How the System Works

### 1️⃣ Data → Features → Training

-   Raw housing data ingestion
-   Repeatable preprocessing and feature engineering pipeline
-   Model training and evaluation
-   **MLflow** experiment tracking (parameters, metrics, artifacts)
-   Model serialization and registration
-   Fully reproducible runs  
    *(same data + same configuration → same model output)*

### 2️⃣ Package the Model as an API

-   Inference wrapped in a **FastAPI** service
-   Model loaded from the **MLflow registry at startup**
-   No hardcoded model artifacts or file paths

**Streamlit UI Layer**

Streamlit acts as a consumer of the FastAPI inference contract:

-   User inputs via forms and sliders
-   Sends HTTP requests to the inference API
-   Displays predictions and optional model metadata

### 3️⃣ Containerization

Docker images include:

-   **FastAPI** application
-   **MLflow** client
-   Model artifact reference or MLflow registry lookup

Images are versioned so that:

-   Model + code + configuration are fully traceable
-   Rollbacks are deterministic and repeatable

### 4️⃣ Kubernetes Deployment

-   Inference pods deployed via **Kubernetes**
-   Services expose **active** and **preview** traffic
-   **ConfigMaps** and **Secrets** inject runtime configuration
-   Resource requests and limits enable sane autoscaling behavior

### 5️⃣ CI with GitHub Actions

-   Validate code via automated tests
-   *(Optional)* train and evaluate models as part of the pipeline
-   Build and publish versioned container images
-   Update GitOps manifests to signal deployment

### 6️⃣ GitOps with Argo CD

-   Kubernetes manifests stored declaratively in **Git**
-   **Argo CD** continuously watches the repository for changes
-   Automatic synchronization of cluster state to Git state
-   Safe rollbacks by reverting commits

### 7️⃣ Controlled Rollouts with Argo Rollouts

-   Replace standard Deployments with **Argo Rollouts**
-   Support **blue/green** and **canary-style** deployments
-   Separate **active** and **preview** services
-   Promote new versions only after validation
-   Enable fast, deterministic rollback on failure

### 8️⃣ Observability (Prometheus + Grafana)

-   Instrument the inference API to emit metrics
-   **Prometheus** scrapes `/metrics` endpoints
-   Track request rate, latency, and error ratios
-   **Grafana** dashboards visualize system behavior

### 9️⃣ Autoscaling with KEDA + HPA

-   Scale inference pods based on real workload signals
-   **KEDA** translates external metrics into scaling signals
-   **HPA** adjusts replica counts automatically
-   Autoscaling driven by latency and request volume (not CPU alone)

## My Additions (What I Layered on Top)

### 1️⃣0️⃣ Blue/Green + Canary-Style Deployments (Argo Rollouts)

This is where the system stops being *“a Kubernetes app”* and starts behaving like **real production ML**.

Instead of replacing pods in place, I introduced **Argo Rollouts** to control **how** new model versions are released.

**What changed:**

-   Replaced standard `Deployment` with an **Argo Rollout**
-   Introduced two logical environments:
    -   **Active** — serving real production traffic
    -   **Preview** — running the new model version

**How it works:**

-   A new model image is deployed to **preview pods**
-   Preview pods are reachable via a **separate service**
-   The active version continues serving production traffic
-   Promotion is **explicit**, not automatic

**This gives me:**

-   Zero-downtime releases
-   The ability to inspect, test, and measure before promotion
-   A clear rollback path if anything looks wrong

>   This is not blue/green for infrastructure —  
>   this is **blue/green for model behavior**.

### 1️⃣1️⃣ Analysis Runs + Controlled Test Traffic

Deploying safely is meaningless unless you can **prove** the new model behaves correctly.

This is where **AnalysisRuns** come in.

**What I added:**

-   **Argo Rollouts AnalysisTemplate**
-   **Prometheus-backed** metrics queries
-   Automated pass/fail promotion gates

**During a rollout:**

-   Argo automatically launches an **AnalysisRun**
-   Prometheus queries evaluate key signals (latency, errors, success rate)
-   If metrics violate constraints:
    -   The rollout halts
    -   The active model remains untouched

**Traffic separation:**

-   Real user traffic → **active** service
-   Synthetic / validation traffic → **preview** service

**This lets me:**

-   Test a new model under load
-   Compare behavior without contaminating production
-   Make promotion a **data-backed decision**, not a guess

### 1️⃣2️⃣ Drift Detection (Post-Deploy Reality Check)

This was the turning point where the system stopped being *“deployment complete”* and became **operational ML**.

Models don’t fail loudly — they **drift quietly**.

So I added explicit drift detection.

**What changed:**

-   Captured a **baseline feature distribution** during training
-   Stored baseline statistics as an **artifact/config**
-   Added a **runtime drift calculation** step during inference

Drift represents the **standardized deviation between training assumptions and live data**.

**How drift is measured:**

-   Incoming inference data is sampled
-   Feature distributions are compared against the baseline
-   A distance-based **drift score** is calculated
-   The score is exposed as a **Prometheus metric**

The system can now answer:

-   *Is the model still seeing the same world it was trained on?*
-   *Has the data changed enough to invalidate predictions?*

**This metric becomes:**

-   A dashboard signal
-   An alerting trigger
-   A future retraining condition

>   This is the difference between **monitoring uptime**  
>   and **monitoring model validity**.

### 1️⃣3️⃣ Closing the Loop (Toward Automated Retraining)

With drift detection in place, the system now has **feedback**.

That enables the next leap:

-   Drift exceeds threshold →
-   Trigger retraining pipeline →
-   Register new model in **MLflow** →
-   Build a new container image →
-   **GitOps** deploy →
-   Canary rollout + analysis →
-   Promote or rollback

I didn’t automate the full loop yet — **on purpose**.

**Why:**

-   I wanted human-in-the-loop visibility
-   I wanted to observe real failure modes
-   I wanted to understand the signals before wiring automation

This is how real systems evolve:

>   **observe → stabilize → automate**

## Why These Additions Matter

These weren’t *“extra features”*.  
They fundamentally changed the nature of the project.

**Before:** Train → deploy → hope  
**After:** Train → validate → deploy → observe → measure → decide

**Key shifts:**

-   Deployment became reversible
-   Metrics became decision gates
-   Drift became a first-class failure mode
-   ML stopped being static artifacts and became a living system

## What I Learned (Why This Mattered)

-   The hardest part of ML isn’t training models — it’s **operating them reliably**
-   **MLflow** makes experimentation and model lineage first-class, not an afterthought
-   **GitOps** makes deployments boring (which is exactly what you want)
-   **Observability isn’t optional** — without metrics, autoscaling is blind
-   **KEDA** is a game-changer when CPU is the wrong scaling signal
-   Debugging **Kubernetes + Prometheus + Argo together** builds real MLOps muscle memory

## Next Upgrades (Because There’s Always a Next)

-   Proper model registry + version promotion (dev → staging → prod)
-   Automated drift detection with retraining triggers
-   CI pipelines that run tests, build images, and push on merge
-   Dashboards and alerts (latency, error rate, saturation, failed scrapes)

## 🎯 Why These Additions Matter

**Before:** Train → deploy → hope  
**After:** Train → validate → deploy → observe → measure → decide

**Key shifts:**

-   Deployment became reversible
-   Metrics became decision gates
-   Drift became a first-class failure mode
-   ML became a living system

***

## 🧠 What I Learned

-   The hardest part of ML isn’t training — it’s **operating models**
-   **MLflow** makes lineage real
-   **GitOps** makes deployments boring (good)
-   **Observability** is mandatory
-   **KEDA** solves the wrong-CPU-metric problem
-   Debugging **Kubernetes + Prometheus + Argo** builds real MLOps muscle

***

## 🔮 Next Upgrades

-   Environment-based model promotion (dev → staging → prod)
-   Automated drift-triggered retraining
-   Fully automated CI/CD pipelines
-   Alerting and SLO-based dashboards

## Closing Thoughts (Why This Project Mattered)

This project wasn’t about learning tools.  
It was about learning **failure modes**.

I broke this system repeatedly:

-   Bad images
-   Broken manifests
-   Silent metric failures
-   Rollouts that stalled
-   Prometheus queries that lied
-   Models that “worked” — but drifted

And each break forced me to understand:

-   Where responsibility actually lives
-   Which signals matter
-   What automation should **not** do
-   How real systems protect themselves

The biggest lesson:

>   **The hardest part of ML isn’t building models —  
>   it’s operating them when nobody is watching.**

This project turned MLOps from a checklist  
into **muscle memory**.

I’ll rebuild it again — cleaner, faster, with fewer mistakes —  
because now I understand *why* each piece exists.

And that’s the difference between  
**following a tutorial**  
and **actually learning MLOps**.
