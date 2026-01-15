# Contributing to PITA

Thank you for your interest in contributing to PITA (Probabilistic Inference Time Algorithms)! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Contributor License Agreement](#contributor-license-agreement)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Questions](#questions)

## Code of Conduct

By participating in this project, you agree to maintain a respectful, inclusive, and professional environment. We expect all contributors to:

- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Contributor License Agreement

### Status: To Be Determined

PITA is dual-licensed under AGPLv3+ and a commercial license. To maintain this dual licensing model, we need clear rights to use contributions in both licensing contexts.

**A formal Contributor License Agreement (CLA) or Developer Certificate of Origin (DCO) process is being developed.**

For now, by contributing to PITA, you acknowledge that:
- Your contributions will be licensed under the same dual-licensing terms as the project (AGPLv3+ and commercial)
- You have the right to submit the contribution under these licenses

### Options Under Consideration

We are evaluating several approaches:

1. **Developer Certificate of Origin (DCO)** - Lightweight process using signed-off commits
2. **Contributor License Agreement (CLA)** - More formal agreement
3. **Copyright Assignment** - Transfer of copyright to COBI, Inc.

**This section will be updated once a decision is made.**

For questions about contribution licensing, contact: sales@cobi-inc.com

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Help us identify and fix issues
- **New features**: Propose and implement new functionality
- **Documentation**: Improve docs, examples, and guides
- **Performance improvements**: Optimize code for speed or efficiency
- **Tests**: Add or improve test coverage
- **Examples**: Create demonstrations and use cases

### Before You Start

1. **Check existing issues**: See if your idea or bug has been discussed
2. **Open an issue**: For significant changes, discuss your approach first
3. **Get feedback**: Ensure your contribution aligns with project goals

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Fork the repository**

   ```bash
   # On GitHub, click "Fork" button
   ```

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR-USERNAME/pita.git
   cd pita
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/cobi-inc-MC/pita.git
   ```

4. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install in development mode**

   ```bash
   # Install core dependencies
   pip install -e .

   # Install development dependencies
   pip install -e ".[dev]"

   # Optional: Install specific backends for testing
   pip install -e ".[vllm]"      # For vLLM backend
   pip install -e ".[llama_cpp]" # For llama.cpp backend
   ```

6. **Verify installation**

   ```bash
   python -c "import pita; print(pita.__version__)"
   ```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Use descriptive branch names:
- `feature/add-new-sampler`
- `fix/vllm-memory-leak`
- `docs/update-installation-guide`

### 2. Make Your Changes

- Write clear, readable code
- Follow our coding standards (see below)
- Add or update tests as needed
- Update documentation if applicable
- Add license headers to new files (see template below)

### 3. Test Your Changes

```bash
# Run tests
pytest

# Run specific test file
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=pita
```

### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with clear message
git commit -m "Add feature: brief description

Longer description of what changed and why.
Fixes #123"
```

**Commit Message Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- First line: Brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues: "Fixes #123" or "Relates to #456"

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Open Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request" button
3. Select base: `main` ← compare: `your-branch`
4. Fill out PR template:
   - Clear description of changes
   - Link to related issues
   - Testing performed

### 7. Code Review

- Address reviewer feedback
- Push additional commits to your branch
- Once approved, maintainers will merge

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md and docs/ as needed
- Include examples in docstrings where helpful

Example:

```python
def sample(context: str, temperature: float = 1.0) -> Output:
    """
    Generate text from the given context.

    Args:
        context: The input text to generate from.
        temperature: Sampling temperature (higher = more random).

    Returns:
        Output object containing generated tokens and metadata.

    Example:
        >>> result = sampler.sample("Hello world", temperature=0.7)
        >>> print(result.tokens)
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific backend tests
pytest tests/inference/base_autoregressive_sampler/test_vllm.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=pita --cov-report=html
```

### Writing Tests

- Add tests for new features
- Ensure existing tests pass
- Aim for good code coverage
- Use descriptive test names
- Test edge cases and error conditions

Example:

```python
def test_sample_with_temperature():
    """Test that sampling with temperature produces valid output."""
    sampler = create_sampler(backend="vllm")
    result = sampler.sample("Test context", temperature=0.7)

    assert result.tokens is not None
    assert len(result.tokens) > 0
    assert result.temperature == 0.7
```

## Questions?

### Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: For licensing questions, contact sales@cobi-inc.com

### Resources

- [README.md](README.md) - Project overview
- [docs/](docs/) - Complete documentation
- [docs/LICENSING-GUIDE.md](docs/LICENSING-GUIDE.md) - Licensing information
- [LICENSE](LICENSE) - AGPLv3+ license text

## Thank You!

Your contributions help make PITA better for everyone. We appreciate your time and effort!

---

**Note**: Contributor licensing terms are still being finalized. Current contributions are accepted under the understanding that they will be dual-licensed (AGPLv3+ and commercial) consistent with the project's licensing model.
