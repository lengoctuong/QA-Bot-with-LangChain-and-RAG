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

### Install and test

```bash
make all
```

### Run app
```python
python main.py
```

### Containerize

```bash
docker build .
docker run -p 8080:8080 --env-file .env CONTAINER_ID
```