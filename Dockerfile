FROM public.ecr.aws/lambda/python:3.10
# FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
# mc-3.12-bullseye: 2.18GB (using gradio from 4.44.1 (or 5.9.1))
# aws-3.10: 1.25GB (using gradio 4.44.1)
# both are good

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["main.py"]
ENTRYPOINT ["python"]