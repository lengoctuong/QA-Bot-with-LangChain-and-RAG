install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# lint:
# 	pylint --disable=R,C 
# 	docker run --rm -i hadolint/hadolint < Dockerfile

test:
	pytest tests
	# pytest -vv --cov= .py

# format:
# 	black *.py
# 	black app/*.py

build: install test
# build: install lint test format