---
title: 🤖 QA Bot with LangChain and RAG
emoji: 🌐🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: main.py
pinned: true
---

# QA Bot with LangChain and RAG ![image](https://github.com/user-attachments/assets/2773b8f7-5fda-408c-ad20-6585bea730b8)

## Install dependencies and test app
```bash
make build
```

## Running

### Run app locally
```python
python main.py
```

### Run using container
```bash
docker build .
docker run -p 8080:8080 --env-file .env CONTAINER_ID
```

## Deploying

### Deploy code repo on HuggingFace Space
Run workflow at ```.github/workflows/hg_main.yml```

### Deploy code (from Github repo) on Azure Web App
Run workflow at ```.github/workflows/az_main.yml```

### Deploy docker image (from Azure Container Registry) Azure Web App

Push image to Azure Container Registry
```bash
az login
az acr login --name CONTAINER_REGISTRY_NAME
docker image tag CONTAINER_ID TAG_REPO
docker push TAG_REPO
```

Create Azure Web App with the image in Azure Container Registry