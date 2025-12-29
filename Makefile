# Makefile

.PHONY: help install dev-install test lint format clean docker-build docker-push terraform-init terraform-apply terraform-destroy k8s-deploy k8s-destroy run-api run-training run-monitoring run-streamlit-dashboard

help:
	@echo "Available commands:"
	@echo "  install                    Install production dependencies"
	@echo "  dev-install                Install development dependencies"
	@echo "  test                       Run tests (not implemented yet)"
	@echo "  lint                       Run linters (not implemented yet)"
	@echo "  format                     Format code (not implemented yet)"
	@echo "  clean                      Clean up temporary files (not implemented yet)"
	@echo "  docker-build-api           Build Docker image for API"
	@echo "  docker-push-api            Push Docker image for API (not implemented yet)"
	@echo "  terraform-init             Initialize Terraform (not implemented yet)"
	@echo "  terraform-apply            Apply Terraform configuration (not implemented yet)"
	@echo "  terraform-destroy          Destroy Terraform resources (not implemented yet)"
	@echo "  k8s-deploy                 Deploy to Kubernetes (not implemented yet)"
	@echo "  k8s-destroy                Destroy Kubernetes resources (not implemented yet)"
	@echo "  run-api                    Run the FastAPI application"
	@echo "  run-training-stage1        Run the Stage 1 training script"
	@echo "  run-monitoring-drift       Run the drift detection script"
	@echo "  run-streamlit-dashboard    Run the Streamlit monitoring dashboard"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt
	# Установка зависимостей для API и мониторинга
	pip install -r requirements-api.txt
	pip install -r requirements-training.txt

# Тесты, линтеры, форматирование, очистка - заглушки
test:
	@echo "Running tests... (not implemented yet)"
	# pytest tests/ -v --cov=api --cov-report=html --cov-report=xml

lint:
	@echo "Running linters... (not implemented yet)"
	# flake8 api/ scripts/ tests/
	# mypy api/ scripts/

format:
	@echo "Formatting code... (not implemented yet)"
	# black api/ scripts/ tests/
	# isort api/ scripts/ tests/

clean:
	@echo "Cleaning up... (not implemented yet)"
	# find . -type d -name "__pycache__" -exec rm -rf {} +
	# find . -type f -name "*.pyc" -delete
	# ... и другие команды очистки

# Docker
docker-build-api:
	docker build -t credit-scoring-api:latest -f deployment/docker/Dockerfile.api .

docker-push-api:
	@echo "Pushing Docker image... (not implemented yet)"
	# docker tag credit-scoring-api:latest your-registry/credit-scoring-api:latest
	# docker push your-registry/credit-scoring-api:latest

# Terraform
terraform-init:
	@echo "Initializing Terraform... (not implemented yet)"
	# cd infrastructure && terraform init

terraform-apply:
	@echo "Applying Terraform configuration... (not implemented yet)"
	# cd infrastructure && terraform apply -auto-approve

terraform-destroy:
	@echo "Destroying Terraform resources... (not implemented yet)"
	# cd infrastructure && terraform destroy -auto-approve

# Kubernetes
k8s-deploy:
	@echo "Deploying to Kubernetes... (not implemented yet)"
	# kubectl apply -f deployment/kubernetes/

k8s-destroy:
	@echo "Destroying Kubernetes resources... (not implemented yet)"
	# kubectl delete -f deployment/kubernetes/

# Запуск приложений
run-api:
	# Убедитесь, что ONNX модель находится в правильном месте для API
	# cp models/credit_scoring_nn.onnx api/models/ # или используйте путь в API
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-training-stage1:
	python scripts/model_training/run_stage1.py

run-monitoring-drift:
	python scripts/monitoring/drift_detection.py

run-streamlit-dashboard:
	streamlit run scripts/streamlit_dashboards/monitoring_dashboard.py
