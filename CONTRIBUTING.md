# Contributing to Lane Detection System

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Road-Lane-Detection-System.git
   cd Road-Lane-Detection-System
   ```
3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-2026.txt
   pip install -r requirements-dev.txt
   ```
4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Follow existing code style (Black, mypy)
   - Add tests for new features
   - Update documentation as needed

3. **Run tests**
   ```bash
   pytest tests/ -v
   mypy lane_detection/
   black --check .
   ruff check .
   ```

4. **Commit your changes**
   ```bash
   git commit -m "feat: add your feature description"
   ```
   
   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation
   - `test:` adding tests
   - `refactor:` code refactoring
   - `perf:` performance improvement
   - `chore:` maintenance

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- **Python**: We use Black (line length 100), isort, mypy, and ruff
- **Type Hints**: All functions should have type hints
- **Docstrings**: Use Google-style docstrings
- **Testing**: Aim for >80% code coverage

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lane_detection --cov-report=html

# Run specific test file
pytest tests/test_detector.py -v

# Skip slow tests during development
pytest tests/ -m "not slow"
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings for all public APIs
- Update type hints
- Add examples for new features

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address review feedback

## Questions?

Feel free to open an issue for discussion before starting major work!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
