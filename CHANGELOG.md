# Changelog

All notable changes to MetaGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2024-11-26

### Added - Core Infrastructure
- Type hints throughout codebase (PEP 484 compliant)
- Pydantic validation for transaction inputs
- Custom exception hierarchy:
  - `MetaGuardError` (base exception)
  - `InvalidTransactionError`
  - `ModelNotFoundError`
  - `ModelLoadError`
  - `ValidationError`
  - `ConfigurationError`
- Comprehensive logging module:
  - `JSONFormatter` for structured logging
  - `ColoredFormatter` for console output
  - `LoggerAdapter` for contextual logging
- Configuration management:
  - `MetaGuardConfig` class with Pydantic validation
  - Environment variable support (METAGUARD_*)
  - Global and instance-level configuration

### Added - ML Models
- Abstract `BaseModel` class for all ML models
- `RandomForestModel` with hyperparameter control
- `XGBoostModel` (optional dependency)
- `EnsembleModel` combining multiple classifiers
- `FeatureEngineer` for derived features:
  - Log-transformed amounts
  - Transactions per day
  - Cyclical hour encoding (sin/cos)
- Benchmark script for model comparison

### Added - API & CLI
- **REST API** (FastAPI):
  - `GET /health` - Health check
  - `POST /detect` - Single transaction detection
  - `POST /detect/batch` - Batch processing (up to 1000)
  - `POST /analyze` - Detailed risk analysis
  - `GET /model/info` - Model information
  - OpenAPI/Swagger documentation
  - CORS support
- **CLI** (Typer + Rich):
  - `metaguard detect` - Detect fraud
  - `metaguard analyze` - Risk analysis
  - `metaguard batch` - Process JSON files
  - `metaguard info` - Model information
  - `metaguard serve` - Start API server
  - JSON output option
  - Beautiful Rich formatting

### Added - DevOps & Production
- Docker support:
  - Multi-stage production Dockerfile
  - Development Dockerfile
  - Docker Compose configuration
- CI/CD with GitHub Actions:
  - Multi-OS testing (Linux, macOS, Windows)
  - Multi-Python testing (3.9-3.12)
  - Security scanning (Bandit, pip-audit)
  - Docker build validation
  - Automated releases to PyPI
- Pre-commit hooks configuration
- Comprehensive Makefile

### Added - Documentation
- Full test suite (282 tests, >90% coverage):
  - Unit tests for all modules
  - Integration tests for API and CLI
  - End-to-end workflow tests
- Sphinx documentation:
  - API reference
  - REST API reference
  - CLI reference
  - Architecture documentation
  - User guide and tutorials
- SECURITY.md with security policy
- CONTRIBUTING.md with contribution guide
- Issue and PR templates

### Changed
- Migrated to src layout structure (`src/metaguard/`)
- Updated to modern pyproject.toml packaging (PEP 517/518)
- Improved batch detection performance
- Enhanced error messages with detailed context
- Updated dependencies for Python 3.9+

### Fixed
- Model compatibility with latest scikit-learn (1.7.x)
- Handling of edge cases in risk calculation
- Zero division protection in risk formulas
- JSON serialization for numpy bool types

## [1.0.0] - 2024-XX-XX

### Added
- Initial release
- `SimpleDetector` class for fraud detection
- `check_transaction()` quick check function
- Random Forest based ML model
- Risk calculation functions:
  - `calculate_risk()`
  - `get_risk_level()`
- Training script for custom models
- Data generation script for synthetic data
- Basic documentation and examples

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.1.0 | TBD | Type hints, validation, logging, tests, documentation |
| 1.0.0 | TBD | Initial MVP release |
