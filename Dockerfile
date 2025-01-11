FROM public.ecr.aws/lambda/python:3.10

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["main.py"]
ENTRYPOINT ["python"]