---
trigger: always_on
---

# Git Auto-Push Workflow

After completing and verifying code updates in this repository:
1. Check `git status` and ensure `.env`, `.venv/`, `data/`, `*.log`, and `*.ex5` remain untracked/ignored (never commit secrets or API keys).
2. Stage modified/new source files, commit with a concise descriptive message, and push to `origin main`.
