<!-- CONTRIBUTING.md — DereInside 社区贡献指南 -->
# Contributing to DereInside

First off, thanks for considering contributing. 🚀

## 📋 Quick Start

```bash
# Clone and install with dev dependencies
git clone https://github.com/derekwang85/derekinside.git
cd derekinside
pip install -e ".[dev,http,ollama]"

# Run tests
python3 -m pytest tests/ -v
```

## 🧭 Where to Start

- **Beginner**: Issues labeled `good first issue`
- **Intermediate**: Issues labeled `help wanted`
- **Expert**: Check the [Roadmap](README.md#-roadmap) for upcoming phases

## 🧩 Code Standards

- **Python**: 3.10+ compatible; type hints required for all public APIs
- **Format**: `ruff format src/` before committing
- **Lint**: `ruff check src/` must pass
- **Tests**: New features require unit tests; bug fixes require regression tests

## 🧠 Architecture Principles

1. **Model is first-class citizen** — every AI endpoint is a named `ModelEndpoint`
2. **Provider is transport** — Driver (Ollama/vLLM/OpenAI) is just how you reach the model
3. **Four-dimensional profile** — every model has `intelligence × cost × speed × quality` metadata
4. **Constraint solving** — pipeline selection is a constraint satisfaction problem, not a fallback chain
5. **Zero-cost evaluation** — all profiling and consensus uses golden data, never external LLMs

## 🧪 Testing

| Layer | Framework | Coverage Target |
|-------|-----------|:---------------:|
| Unit | pytest | 80%+ |
| Integration | pytest + DB | 60%+ |
| Benchmark | LongMemEval | N/A |

```bash
# Unit tests
python3 -m pytest tests/ -v --ignore=tests/test_integration.py

# Integration tests (requires PostgreSQL + pgvector)
python3 -m pytest tests/test_integration.py -v

# LongMemEval benchmark
python3 scripts/long_mem_eval.py
```

## 📝 Commit Message Convention

Follow the three-track prefix system:

- `feat:` — new feature (minor version bump)
- `fix:` — bug fix (patch version bump)
- `docs:` — documentation only
- `refactor:` — code change without feature/fix
- `test:` — test additions or changes
- `chore:` — build/config/tooling

## 🔄 PR Workflow

1. Fork and create a feature branch from `main`
2. Write code + tests
3. Run `ruff check src/ && ruff format --check src/`
4. Open a PR with:
   - Clear title using prefix convention
   - Description of what changed and why
   - Closes #issue-number if applicable
   - Ripple analysis checklist
5. At least one approval required before merge

## 💬 Getting Help

- Open a [Discussion](https://github.com/derekwang85/derekinside/discussions)
- Check existing [Issues](https://github.com/derekwang85/derekinside/issues)
