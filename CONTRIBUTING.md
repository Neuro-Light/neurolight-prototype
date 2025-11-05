# Contributing to NeuroLight

Thank you for your interest in contributing to NeuroLight! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate in all interactions
- Accept constructive criticism gracefully
- Focus on what is best for the community and the project
- Show empathy towards other community members

## Getting Started

Before you begin contributing:

1. **Familiarize yourself with the project** - Read the [README.md](README.md) to understand NeuroLight's purpose and architecture
2. **Check existing issues** - Browse [open issues](https://github.com/Neuro-Light/neurolight-prototype/issues) to see if your idea or bug has already been reported
3. **Join the discussion** - Comment on issues you're interested in working on to avoid duplicate efforts

## Development Setup

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Fork the repository on GitHub

2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/neurolight-prototype.git
cd neurolight-prototype
```

3. Add the upstream repository:
```bash
git remote add upstream https://github.com/Neuro-Light/neurolight-prototype.git
```

4. Install dependencies:
```bash
uv sync
```

5. Run the application to ensure everything works:
```bash
uv run python src/main.py
```

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Screenshots or error messages (if applicable)
- Your environment details (OS, Python version, etc.)
- Sample data or experiments that trigger the bug (if applicable)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- A clear description of the feature
- Use cases explaining why this feature would be valuable
- Potential implementation approach (if you have ideas)
- Any relevant mockups or examples from other applications

### Code Contributions

1. **Create a new branch** for your work:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

2. **Make your changes** following our [coding standards](#coding-standards)

3. **Test your changes** thoroughly

4. **Commit your changes** with clear, descriptive messages:
```bash
git commit -m "Add feature: brief description of what you added"
```

5. **Keep your fork updated**:
```bash
git fetch upstream
git rebase upstream/main
```

6. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

7. **Submit a Pull Request** (see [Pull Request Process](#pull-request-process))

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Code Organization

- Keep functions focused and single-purpose
- Add docstrings to all public functions and classes
- Use descriptive comments for complex logic
- Follow the existing project structure:
  - `src/ui/` - UI components
  - `src/core/` - Core functionality
  - `src/utils/` - Utility functions

### Example Function Documentation

```python
def load_image_stack(path: str, validate: bool = True) -> ImageStack:
    """
    Load a TIF image stack from the specified path.
    
    Args:
        path: Absolute or relative path to the TIF file or directory
        validate: Whether to validate image dimensions and format
        
    Returns:
        ImageStack object containing the loaded images
        
    Raises:
        FileNotFoundError: If the path does not exist
        ValueError: If images fail validation
    """
    pass
```

### UI Guidelines

- Maintain consistency with existing UI patterns
- Ensure all UI elements are accessible and clearly labeled
- Use Qt signals/slots for event handling
- Keep UI logic separate from business logic

## Testing Guidelines

We use `pytest` for testing. All contributions should include tests where applicable.

### Running Tests

```bash
uv sync --all-extras  # Install test dependencies
pytest tests/
```

### Writing Tests

- Create test files in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases

Example:

```python
def test_experiment_creation_with_valid_data():
    """Test that experiments are created successfully with valid metadata."""
    # Arrange
    experiment_data = {
        "name": "Test Experiment",
        "description": "Test description"
    }
    
    # Act
    experiment = ExperimentManager.create_experiment(experiment_data)
    
    # Assert
    assert experiment.name == "Test Experiment"
    assert experiment.description == "Test description"
```

### Test Coverage

- Aim for meaningful test coverage of your changes
- Focus on testing critical paths and edge cases
- Test error handling and validation logic

## Pull Request Process

1. **Ensure your PR**:
   - Follows the coding standards
   - Includes tests for new functionality
   - Updates documentation if needed
   - Passes all existing tests

2. **PR Title**: Use a clear, descriptive title
   - `Feature: Add YOLOv8 detection integration`
   - `Fix: Resolve image loading crash with corrupted TIFs`
   - `Docs: Update installation instructions`

3. **PR Description**: Include:
   - Summary of changes
   - Related issue numbers (e.g., "Closes #123")
   - Testing performed
   - Screenshots (for UI changes)
   - Breaking changes (if any)

4. **Review Process**:
   - A maintainer will review your PR
   - Address any requested changes
   - Once approved, your PR will be merged

## Issue Guidelines

### Before Creating an Issue

- Search existing issues to avoid duplicates
- Check if your issue might be a question better suited for discussions

### Issue Templates

**Bug Report:**
```
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. Windows 11, macOS 14]
 - Python version: [e.g. 3.10]
 - NeuroLight version: [e.g. commit hash]
```

**Feature Request:**
```
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request.
```

## Areas for Contribution

### Issue Labels and What They Mean

We label issues that would be good for a first time contributor as [**`good first issue`**](https://github.com/Neuro-Light/neurolight-prototype/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). These usually do not require significant experience with Python, PySide6, or the NeuroLight codebase.

We label issues that we think are a good opportunity for subsequent contributions as [**`help wanted`**](https://github.com/Neuro-Light/neurolight-prototype/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22). These require varying levels of experience with Python and NeuroLight. Often, we want to accomplish these tasks but do not have the resources to do so ourselves.

**Please check in with us before starting work on an issue that has not been labeled as appropriate for community contribution.** We're happy to receive contributions for other issues, but it's important to make sure we have consensus on the solution to the problem first.

Outside of issues with the labels above, issues labeled as [**`bug`**](https://github.com/Neuro-Light/neurolight-prototype/issues?q=is%3Aissue+is%3Aopen+label%3Abug) are the best candidates for contribution. In contrast, issues labeled with **`needs-decision`** or **`needs-design`** are not good candidates for contribution. Please do not open pull requests for issues with these labels.

**Please do not open pull requests for new features without prior discussion.** While we appreciate exploration of new features, we will almost always close these pull requests immediately. Adding a new feature to NeuroLight creates a long-term maintenance burden and requires strong consensus from the NeuroLight team before it is appropriate to begin work on an implementation.

### Good First Issues

Look for issues tagged with `good first issue` - these are great for newcomers and typically include:

- Documentation improvements
- Adding code comments and docstrings
- Simple UI text or label improvements
- Example experiment templates
- Minor bug fixes with clear solutions

## Questions?

If you have questions about contributing:

- Open a discussion on GitHub Discussions
- Comment on relevant issues
- Reach out to maintainers

## License

By contributing to NeuroLight, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to NeuroLight!** Your efforts help advance neuroscience research tools for scientists worldwide. 🧠✨
