# 🚨 Scam Alert Detection AI Agent

**Scam Alert Detection AI Agent** is an AI-powered system that flags potentially fraudulent profiles on dating platforms.
It combines **machine learning** (Random Forest, XGBoost) and **deep learning** (AnomalyNet) models with a **Retrieval-Augmented Generation (RAG)** pipeline to provide **both predictions and explanations**.
The system integrates with **AWS S3, DynamoDB, Lambda, and Step Functions**, and offers a **Streamlit UI** for human review.

---

## 📚 Table of Contents

1. [Features](#-features)
2. [Tech Stack](#-tech-stack)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Configuration](#-configuration)
6. [Usage](#-usage)
7. [Training Models](#-training-models)
8. [Deployment](#-deployment)
9. [Running Tests](#-running-tests)
10. [Contributing](#-contributing)
11. [License](#-license)

---

## ✨ Features

* **Multi-Model Scam Detection** – Combines Random Forest, XGBoost, and AnomalyNet for robust detection.
* **RAG Explainability** – Retrieves similar profiles and context for each flagged case.
* **AWS Integration** – Uses S3 for model/config storage, DynamoDB for profile storage, Lambda for inference, and Step Functions for pipeline orchestration.
* **Streamlit UI** – Interactive dashboard for profile review and investigation.
* **Configurable Thresholds** – Flag profiles based on custom anomaly score thresholds.
* **Logging & Monitoring** – Full logs for debugging and audit trails.

---

## 🖥 Tech Stack

**AI Models**

* Scikit-learn (Random Forest)
* XGBoost
* PyTorch (AnomalyNet)
* Sentence Transformers (RAG embeddings)

**Data & Infrastructure**

* AWS S3 – Model/config storage
* AWS DynamoDB – Profile storage
* AWS Lambda – Serverless inference & RAG
* AWS Step Functions – Workflow orchestration

**Frontend**

* Streamlit – UI for profile review

**Other**

* Pandas, NumPy – Data processing
* PyYAML – Config management
* Joblib – Model persistence

---

## 📂 Project Structure

```
scam_alert_detection/
├── streamlit/secrets.toml           # Streamlit secrets (API keys, AWS creds)
├── config.yaml                      # Model + AWS configuration
├── Dockerfile                       # Streamlit UI & local testing container
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ignore secrets, models, data, envs
├── secrets/
│   └── aws_credentials.json         # AWS Lambda/DynamoDB credentials
├── data/
│   ├── Profiles.csv
│   ├── BlockedUsers.csv
│   ├── DeclinedUsers.csv
│   ├── DeletedUsers.csv
│   ├── ReportedUsers.csv
├── models/
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── pytorch_model.pth
│   ├── profile_embeddings.pkl
├── src/
│   ├── data_loader.py
│   ├── preprocess_data.py
│   ├── train_anomaly_net.py
│   ├── predict.py
│   ├── rag_pipeline.py
│   ├── lambda_preprocess.py
│   ├── lambda_predict.py
│   ├── lambda_rag.py
├── ui/
│   └── streamlit_app.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocess.py
│   ├── test_predict.py
│   ├── test_rag_pipeline.py
├── .github/workflows/
│   └── ci.yml
├── step_functions/
│   └── rag_workflow.json
└── README.md
```

---

## 🛠 Installation

```bash
git clone https://github.com/your-repo/scam-alert-detection.git
cd scam_alert_detection
pip install -r requirements.txt
```

---

## ⚙️ Configuration

* **Streamlit secrets** (`streamlit/secrets.toml`):

```toml
OPENAI_API_KEY="your_openai_api_key"
AWS_ACCESS_KEY_ID="your_access_key"
AWS_SECRET_ACCESS_KEY="your_secret_key"
AWS_REGION="your_region"
```

* **AWS credentials** (`secrets/aws_credentials.json`):

```json
{
  "aws_access_key_id": "...",
  "aws_secret_access_key": "...",
  "region_name": "..."
}
```

* **Model & AWS config** (`config.yaml`):

```yaml
s3_bucket: "scam-alert-detection-models"
dynamodb_tables:
  profiles: "Profiles"
  blocked: "BlockedUsers"
  embeddings: "ProfileEmbeddings"
anomaly_threshold: 0.7
```

---

## 🚀 Usage

**Run locally with Streamlit:**

```bash
streamlit run ui/streamlit_app.py
```

Then open **[http://localhost:8501](http://localhost:8501)**.

**Profile Review Flow:**

1. Upload/select a profile.
2. Model predicts anomaly score.
3. If score > threshold, profile is flagged.
4. RAG pipeline retrieves similar profiles and explanations.

---

## 🧠 Training Models

```bash
python src/preprocess_data.py
python src/train_anomaly_net.py
```

* Place trained Random Forest and XGBoost models in `models/`.

---

## ☁️ Deployment

**AWS Setup**

1. Create S3 bucket & DynamoDB tables.
2. Upload `config.yaml` to S3.
3. Deploy Lambda functions:

   * `scam-alert-preprocess`
   * `scam-alert-predict`
   * `scam-alert-rag`

**CI/CD with GitHub Actions**

* Set repository secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
* Push to `main` to trigger `.github/workflows/ci.yml`.

**Step Functions**

* Deploy `step_functions/rag_workflow.json` for RAG orchestration.

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 🤝 Contributing

1. Fork & create feature branch:
   `git checkout -b feature/your-feature`
2. Commit changes:
   `git commit -m "Add new feature"`
3. Push & open PR.

---

## 📜 License

MIT License.

---

