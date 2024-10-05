# 🔬 Kidney Tumor Classification ML Project


## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [MLflow Integration](#mlflow-integration)
- [DVC (Data Version Control)](#dvc-data-version-control)
- [AWS Deployment](#aws-deployment)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Project Overview

This cutting-edge machine learning project focuses on the critical task of kidney tumor classification. Leveraging state-of-the-art deep learning techniques, our model analyzes medical imaging data to accurately classify kidney tumors, potentially aiding in early detection and diagnosis.

### 🎯 Objective
To develop a robust, scalable, and accurate machine learning model for classifying kidney tumors, deployable in real-world medical settings.

### 💡 Key Features

- **High Accuracy**: Utilizes advanced CNN architectures for precise tumor classification.
- **End-to-End Solution**: From data preprocessing to model deployment, all in one package.
- **Scalability**: Designed to handle large datasets efficiently.
- **MLflow Integration**: For comprehensive experiment tracking and model versioning.
- **DVC Pipeline**: Ensures reproducibility and efficient data/model versioning.
- **Containerized Deployment**: Docker-based deployment for consistency across environments.
- **Cloud-Ready**: Seamless deployment on AWS infrastructure.

## 🛠 Tech Stack

- **Python 3.8**
- **TensorFlow/Keras**: For building and training the deep learning model
- **MLflow**: Experiment tracking and model management
- **DVC**: Data and pipeline versioning
- **Docker**: Containerization
- **AWS**: Cloud deployment (EC2, ECR)
- **GitHub Actions**: CI/CD pipeline

## 📂 Project Structure

```
kidney-tumor-classification/
│
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data_processing/
│   ├── model/
│   ├── training/
│   └── evaluation/
├── tests/
├── config/
├── .github/workflows/
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8
- Conda

### Installation

1. Clone the repository:
   ```bash
   [git clone https://github.com/arpansethi30/Kidney-Disease-Classification.git
   cd Kidney-Disease-Classification
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n kidney-tumor python=3.8 -y
   conda activate kidney-tumor
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
python main.py
```

## 📊 MLflow Integration

We use MLflow for comprehensive experiment tracking and model management.

To view the MLflow UI:
```bash
mlflow ui
```

Set up DagsHub for remote tracking:
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/your-username/kidney-tumor-classification.mlflow
export MLFLOW_TRACKING_USERNAME=your-username
export MLFLOW_TRACKING_PASSWORD=your-token
```

## 🔄 DVC (Data Version Control)

DVC is used for efficient data and pipeline versioning.

Key commands:
- `dvc init`: Initialize DVC in your project
- `dvc repro`: Reproduce the entire pipeline
- `dvc dag`: View the pipeline structure

## ☁️ AWS Deployment

Our project is deployed on AWS using GitHub Actions for CI/CD.

### Setup Steps:
1. Create an IAM user with EC2 and ECR access
2. Set up an ECR repository
3. Launch and configure an EC2 instance
4. Set up GitHub Secrets:
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_REGION
   - AWS_ECR_LOGIN_URI
   - ECR_REPOSITORY_NAME

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by Arpan.
</p>
