# Copilot Repository Instructions for Microgrid-load-smoothing

## 🎯 Project Goals
This project focuses on microgrid load smoothing, renewable integration, and energy scheduling.
The primary goal of Copilot tasks is to **restore runnability**, **reduce technical debt**, and **improve maintainability** without altering research logic.

---

## 🧠 Task Priorities
1. ✅ Ensure the project can **run successfully** end-to-end with sample data.
2. 🧩 Identify and fix dependency or version issues.
3. 🧪 Add smoke tests or minimal unit tests under `tests/`.
4. 📘 Update README and generate `REPORT.md` summarizing findings.
5. ⚙️ Suggest code structure improvements only after confirming it runs.

---

## 🚫 Do Not
- Modify algorithmic or model logic unless explicitly requested.
- Remove research-specific experimental scripts or results.
- Commit any large binary datasets or generated plots.

---

## 🧰 Guidelines for Refactor / Fix Tasks
- Follow **PEP8** and docstring conventions (Google style).
- Use explicit imports (no wildcard `from x import *`).
- Prefer logging over print.
- Keep compatibility with Python 3.9+.
- If external tools (e.g., PyTorch, TensorFlow) are used, pin minimal working versions.

---

## 🧪 Validation Rules
When asked to “make project runnable,” ensure:
- `pip install -r requirements.txt` succeeds.
- `python main.py` or equivalent script runs without fatal errors.
- A minimal smoke test passes.
- CI workflow status = ✅.

---

## 🗒️ Output Requirements
For each task, generate or update:
- `REPORT.md`: summary of what was fixed and remaining issues.
- `tests/test_smoke.py`: if missing, create a simple test that imports main modules.
- Updated README if new setup steps are added.

---

## 🧩 Example Tasks
- `Analyze repository and report outdated dependencies`
- `Add smoke test to verify that simulation runs on sample data`
- `Refactor scripts/utils.py to remove duplicate code`
- `Pin versions in requirements.txt and update README`

---

*(Last updated by zqy2023, Oct 2025)*
