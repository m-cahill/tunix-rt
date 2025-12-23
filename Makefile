.PHONY: help install install-backend install-frontend install-e2e test test-backend test-frontend test-e2e lint lint-backend format format-backend docker-up docker-down docker-logs clean

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: install-backend install-frontend install-e2e  ## Install all dependencies

install-backend:  ## Install backend dependencies
	cd backend && python -m pip install -e ".[dev]"

install-frontend:  ## Install frontend dependencies
	cd frontend && npm ci

install-e2e:  ## Install E2E test dependencies
	cd e2e && npm ci && npx playwright install chromium --with-deps

# Testing targets
test: test-backend test-frontend  ## Run all tests (excluding E2E)

test-backend:  ## Run backend tests with coverage
	cd backend && pytest --cov=tunix_rt_backend --cov-branch --cov-report=term --cov-report=json:coverage.json -v
	cd backend && python tools/coverage_gate.py

test-frontend:  ## Run frontend tests
	cd frontend && npm run test

test-e2e:  ## Run E2E tests (mock mode)
	cd e2e && REDIAI_MODE=mock npx playwright test

test-e2e-real:  ## Run E2E tests (real mode, requires RediAI)
	cd e2e && REDIAI_MODE=real npx playwright test

# M4: Local E2E with full infrastructure
e2e:  ## Run E2E tests with full setup (postgres + migrations + playwright)
	@echo "Starting Postgres..."
	docker compose up -d postgres
	@echo "Waiting for Postgres to be ready..."
	@timeout 30 sh -c 'until docker compose exec -T postgres pg_isready -U postgres; do sleep 1; done' || (echo "Postgres failed to start" && exit 1)
	@echo "Running migrations..."
	cd backend && alembic upgrade head
	@echo "Running E2E tests (Playwright will start backend + frontend)..."
	cd e2e && REDIAI_MODE=mock npx playwright test
	@echo "E2E tests complete. Postgres left running for iteration. Use 'make e2e-down' to stop."

e2e-down:  ## Stop E2E infrastructure (postgres)
	@echo "Stopping Postgres..."
	docker compose down

# Linting and formatting targets
lint: lint-backend  ## Run all linters

lint-backend:  ## Run backend linting and type checking
	cd backend && ruff check .
	cd backend && mypy tunix_rt_backend

format: format-backend  ## Format all code

format-backend:  ## Format backend code
	cd backend && ruff format .

# Database migration targets
db-upgrade:  ## Run database migrations (upgrade to head)
	cd backend && alembic upgrade head

db-downgrade:  ## Rollback last database migration
	cd backend && alembic downgrade -1

db-revision:  ## Create a new migration (usage: make db-revision msg="description")
	cd backend && alembic revision --autogenerate -m "$(msg)"

db-current:  ## Show current database revision
	cd backend && alembic current

db-history:  ## Show migration history
	cd backend && alembic history

# Docker targets
docker-up:  ## Start Docker Compose services
	docker compose up -d

docker-down:  ## Stop Docker Compose services
	docker compose down

docker-logs:  ## Show Docker Compose logs
	docker compose logs -f

docker-restart:  ## Restart Docker Compose services
	docker compose restart

# Build targets
build-frontend:  ## Build frontend for production
	cd frontend && npm run build

# Clean targets
clean:  ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	cd backend && rm -rf *.egg-info dist build coverage.json .coverage 2>/dev/null || true
	cd frontend && rm -rf dist node_modules/.cache 2>/dev/null || true
	cd e2e && rm -rf playwright-report test-results 2>/dev/null || true
