# 🛠️ Development Tools Guide

This document explains all the development tools used in the NeuroLight project, how to configure them, and how to use them effectively.

---

## Table of Contents

- [Overview](#overview)
- [Ruff - Linter & Formatter](#ruff---linter--formatter)
- [MyPy - Type Checker](#mypy---type-checker)
- [Pyrefly - Type Checker](#pyrefly---type-checker)
- [Pytest - Testing Framework](#pytest---testing-framework)
- [Hatchling - Build Backend](#hatchling---build-backend)
- [uv - Package Manager](#uv---package-manager)
- [Quick Reference](#quick-reference)

---

## Overview

NeuroLight uses a modern Python development toolchain to ensure code quality, type safety, and maintainability:

| Tool | Purpose | Version |
|------|---------|---------|
| **Ruff** | Fast Python linter and formatter | ≥0.14.3 |
| **MyPy** | Static type checker | ≥1.0.0 |
| **Pyrefly** | Additional type checking | Configured |
| **Pytest** | Testing framework | ≥7.0.0 |
| **Hatchling** | Build backend | Latest |
| **uv** | Package manager | Latest |

---

## Ruff - Linter & Formatter

**Ruff** is an extremely fast Python linter and code formatter written in Rust. It replaces multiple tools like Flake8, isort, and Black with a single, fast tool.

In this project, we use `uvx` to run Ruff. `uvx` is uv's tool runner that automatically downloads and executes Python tools in isolated environments without requiring global installation. This ensures everyone uses the same version and keeps your system clean.

### Installation

Ruff is included in the development dependencies, but you don't need to install it globally. Use `uvx` to run ruff without installation:

```bash
# Sync dev dependencies (installs ruff and mypy)
# This is recommended for IDE integration and type checking
uv sync --group dev

# Or just use uvx for ruff - no installation needed!
# uvx automatically downloads and runs ruff
# Note: MyPy still needs to be installed for type checking
```

### Configuration

Ruff configuration can be added to `pyproject.toml` under `[tool.ruff]`. Here's a comprehensive configuration example:

```toml
[tool.ruff]
# Set the maximum line length
line-length = 100

# Enable specific rule sets
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "ARG",    # flake8-unused-arguments
    "PIE",    # flake8-pie
    "PT",     # flake8-pytest-style
    "RET",    # flake8-return
    "RUF",    # Ruff-specific rules
]

# Ignore specific rules
ignore = [
    "E501",   # Line too long (handled by formatter)
    "B008",   # Do not perform function calls in argument defaults
    "ARG001", # Unused function argument
    "ARG002", # Unused method argument
]

# Exclude files and directories
exclude = [
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "*.egg-info",
    "node_modules",
]

# Allow autofix for all enabled rules
fixable = ["ALL"]
unfixable = []

# Target Python version
target-version = "py39"

[tool.ruff.format]
# Use double quotes for strings
quote-style = "double"
# Indent with spaces
indent-style = "space"
# Respect magic trailing comma
skip-magic-trailing-comma = false
# Automatically detect the appropriate line ending
line-ending = "auto"

[tool.ruff.lint]
# Enable specific rule categories
select = ["ALL"]
# But ignore some rules
ignore = [
    "E501",   # Line too long
    "B008",   # Function calls in defaults
    "ARG001", # Unused function argument
    "ARG002", # Unused method argument
    "PLR0913", # Too many arguments
    "PLR2004", # Magic value used in comparison
]

[tool.ruff.lint.per-file-ignores]
# Ignore specific rules in test files
"tests/*" = ["ARG", "S101"]
# Ignore import sorting in __init__.py files
"__init__.py" = ["I"]

[tool.ruff.lint.isort]
# Configure import sorting
known-first-party = ["src"]
force-single-line = false
split-on-trailing-comma = true
```

### Usage

#### Check for issues (without fixing)

```bash
# Check all files
uvx ruff check .

# Check specific file or directory
uvx ruff check src/main.py
uvx ruff check src/

# Show source code for violations
uvx ruff check . --show-source
```

#### Auto-fix issues

```bash
# Fix all auto-fixable issues
uvx ruff check --fix .

# Fix and format code
uvx ruff check --fix .
uvx ruff format .
```

#### Format code

```bash
# Format all files
uvx ruff format .

# Format specific file
uvx ruff format src/main.py

# Check formatting without making changes
uvx ruff format --check .
```

#### Common Ruff Commands

```bash
# Run both linting and formatting
uvx ruff check --fix . && uvx ruff format .

# Show statistics
uvx ruff check . --statistics

# Output in different formats
uvx ruff check . --output-format=json
uvx ruff check . --output-format=github  # For GitHub Actions

# Watch mode (requires ruff-lsp)
uvx ruff check . --watch
```

### Integration with IDEs

**VS Code / Cursor:**
- Install the "Ruff" extension
- It will automatically use your `pyproject.toml` configuration

**PyCharm:**
- Ruff can be configured as an external tool
- Or use the Ruff plugin from the JetBrains marketplace

---

## MyPy - Type Checker

**MyPy** is a static type checker for Python that helps catch type-related errors before runtime.

### Installation

MyPy is included in the development dependencies:

```bash
uv sync --group dev
```

This installs both Ruff and MyPy along with other dev tools.

### Configuration

Add MyPy configuration to `pyproject.toml`:

```toml
[tool.mypy]
# Python version to type check against
python_version = "3.9"

# Enable strict type checking
strict = false

# Show error codes
show_error_codes = true

# Warn about unused ignores
warn_unused_ignores = true

# Warn about redundant casts
warn_redundant_casts = true

# Warn about unused configs
warn_unused_configs = true

# Check untyped definitions
check_untyped_defs = true

# Disallow untyped calls
disallow_untyped_calls = false

# Disallow untyped definitions
disallow_untyped_defs = false

# Disallow incomplete type stubs
disallow_incomplete_defs = false

# Check that functions have type annotations
disallow_untyped_decorators = false

# Warn about missing return statements
warn_return_any = true

# Warn about unused 'type: ignore' comments
warn_unused_ignores = true

# Warn about no return statements
warn_no_return = true

# Follow imports
follow_imports = "normal"

# Ignore missing imports
ignore_missing_imports = false

# Show column numbers
show_column_numbers = true

# Show error context
show_error_context = true

# Pretty output
pretty = true

# Color output
color_output = true

# Incremental mode (faster for large codebases)
incremental = true

# Cache directory
cache_dir = ".mypy_cache"

# Exclude patterns
exclude = [
    "build/",
    "dist/",
    ".venv/",
    "venv/",
    "__pycache__/",
]

[[tool.mypy.overrides]]
# Allow untyped calls in test files
module = "tests.*"
disallow_untyped_calls = false
disallow_untyped_defs = false
```

### Usage

```bash
# Type check all files
mypy src/

# Type check specific file
mypy src/main.py

# Show error codes
mypy src/ --show-error-codes

# Stub files only (for libraries)
mypy src/ --check-untyped-defs

# Ignore missing imports (useful for third-party libraries)
mypy src/ --ignore-missing-imports

# Follow imports
mypy src/ --follow-imports=normal
```

### Type Checking Best Practices

1. **Use type hints everywhere:**
   ```python
   def process_image(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
       """Process an image with the given threshold."""
       return image > threshold
   ```

2. **Use `typing` module for complex types:**
   ```python
   from typing import List, Optional, Dict, Tuple
   
   def get_experiments() -> List[Experiment]:
       ...
   ```

3. **Use `# type: ignore` sparingly:**
   ```python
   # Only when absolutely necessary
   result = some_function()  # type: ignore[assignment]
   ```

---

## Pyrefly - Type Checker

**Pyrefly** is an additional type checker configured in the project. It provides complementary type checking to MyPy.

### Configuration

Pyrefly is configured in `pyproject.toml`:

```toml
[tool.pyrefly]
# Files to include
project-includes = ["**/*.py", "**/*.pyi"]

# Files to exclude
project-excludes = [
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/.venv/**"
]

# Use .gitignore and other ignore files
use-ignore-files = true

# Target Python version
python-version = "3.9"

# Search path
search-path = ["."]
```

### Usage

**Command Line:**

```bash
# Check all files
pyrefly check

# Check specific directory
pyrefly check src/

# Check specific file
pyrefly check src/main.py

# With error summary
pyrefly check --summarize-errors
```

**IDE Integration:**

Pyrefly can also run automatically as part of your IDE. For VS Code/Cursor:
- Install the Pyrefly extension from the marketplace
- It will automatically check your code as you type
- Errors will appear in the Problems panel

**Verifying It's Working:**

Create a test file to verify Pyrefly is working:

```python
# test_pyrefly.py
def test_function(x: int) -> str:
    return x  # This should trigger a type error
```

Run:
```bash
pyrefly check test_pyrefly.py
```

If Pyrefly is working, it should report that `x` (an `int`) cannot be returned where a `str` is expected.

---

## Pytest - Testing Framework

**Pytest** is the testing framework used for unit and integration tests.

### Installation

Pytest is included in the test dependencies:

```bash
uv sync --extra test
```

### Configuration

Add pytest configuration to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
# Test discovery patterns
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

# Test paths
testpaths = ["tests"]

# Output options
addopts = [
    "-v",                    # Verbose output
    "--strict-markers",      # Strict marker checking
    "--tb=short",           # Short traceback format
    "--cov=src",            # Coverage for src directory
    "--cov-report=term-missing",  # Show missing lines
    "--cov-report=html",    # Generate HTML report
]

# Markers
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

# Minimum Python version
minversion = "7.0"

# Timeout for tests (in seconds)
timeout = 300

# Warnings
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]
```

### Usage

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_experiment_manager.py

# Run specific test function
pytest tests/test_experiment_manager.py::test_create_experiment

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Writing Tests

```python
import pytest
from src.core.experiment_manager import Experiment, ExperimentManager

def test_experiment_creation():
    """Test that experiments can be created successfully."""
    experiment = Experiment(
        name="Test Experiment",
        description="A test experiment"
    )
    assert experiment.name == "Test Experiment"
    assert experiment.description == "A test experiment"

@pytest.fixture
def sample_experiment():
    """Fixture providing a sample experiment."""
    return Experiment(name="Sample", description="Sample experiment")

def test_experiment_with_fixture(sample_experiment):
    """Test using a fixture."""
    assert sample_experiment.name == "Sample"
```

---

## Hatchling - Build Backend

**Hatchling** is the build backend used to package and distribute the project.

### Configuration

Hatchling is configured in `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

### Usage

```bash
# Build the package
uv build

# Build source distribution
uv build --sdist

# Build wheel
uv build --wheel
```

---

## uv - Package Manager

**uv** is the fast Python package manager used in this project.

### Common Commands

```bash
# Install all dependencies
uv sync

# Install with dev dependencies (ruff, mypy, etc.)
# Uses [dependency-groups] dev
uv sync --group dev

# Install with test dependencies (pytest, pytest-cov)
# Uses [project.optional-dependencies] test
uv sync --extra test

# Install with all extras (test dependencies)
uv sync --all-extras

# Run a command in the virtual environment
uv run neurolight
uv run pytest

# Run ruff via uvx (no installation needed)
uvx ruff check .

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --group dev package-name

# Remove a dependency
uv remove package-name

# Update dependencies
uv sync --upgrade

# Show installed packages
uv pip list

# Lock dependencies
uv lock
```

---

## Quick Reference

### Pre-commit Checklist

Before committing code, run these commands:

```bash
# 1. Format code
uvx ruff format .

# 2. Lint and auto-fix
uvx ruff check --fix .

# 3. Type check
uv run mypy src/

# 4. Run tests
uv run pytest

# 5. Check everything passes
uvx ruff check . && uv run mypy src/ && uv run pytest
```

### Recommended Workflow

1. **Make changes** to your code
2. **Format** with `uvx ruff format .`
3. **Lint and fix** with `uvx ruff check --fix .`
4. **Type check** with `uv run mypy src/`
5. **Test** with `uv run pytest`
6. **Commit** when all checks pass

### IDE Integration

**VS Code / Cursor:**
- Install extensions: Ruff, Pylance (for MyPy), Python Test Explorer
- Configure settings:
  ```json
  {
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "none",
    "[python]": {
      "editor.defaultFormatter": "charliermarsh.ruff",
      "editor.formatOnSave": true,
      "editor.codeActionsOnSave": {
        "source.fixAll": true
      }
    }
  }
  ```

**PyCharm:**
- Install plugins: Ruff, MyPy
- Configure Ruff as external tool
- Enable MyPy inspection

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: astral-sh/setup-uv@v1
      - run: uv sync --group dev
      - run: uvx ruff check .
      - run: uvx ruff format --check .
      - run: uv run mypy src/
      - run: uv run pytest
```

---

## Troubleshooting

### Ruff Issues

**Problem:** Ruff not finding configuration
- **Solution:** Ensure `[tool.ruff]` is in `pyproject.toml` at the root

**Problem:** Ruff ignoring files it shouldn't
- **Solution:** Check `exclude` patterns in `[tool.ruff]`

### MyPy Issues

**Problem:** MyPy can't find imports
- **Solution:** Add `--ignore-missing-imports` or configure `ignore_missing_imports = true`

**Problem:** MyPy too strict
- **Solution:** Adjust `strict = false` and configure specific rules

### Pytest Issues

**Problem:** Tests not discovered
- **Solution:** Check `testpaths` and `python_files` in `[tool.pytest.ini_options]`

**Problem:** Import errors in tests
- **Solution:** Ensure `src` is in Python path or use `pytest --import-mode=importlib`

---

## Additional Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Hatchling Documentation](https://hatch.pypa.io/)
- [uv Documentation](https://github.com/astral-sh/uv)

---

**Happy coding brainiacs! 🚀**

