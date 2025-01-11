install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C 
	#docker run --rm -i hadolint/hadolint < Dockerfile

test:
	pytest -vv --cov= .py

format:
	black *.py
	black app/*.py

build: install lint test format