from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _result(level: str, label: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"[{level}] {label}{suffix}")


def _git_executable() -> Path | None:
    if value := shutil.which("git"):
        return Path(value)
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        return None
    candidates = sorted(
        Path(local_app_data).glob(
            "GitHubDesktop/app-*/resources/app/git/cmd/git.exe"
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _run_git(git: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(git), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def check_local_environment() -> int:
    failures = 0
    if (PROJECT_ROOT / "pyproject.toml").is_file():
        _result("PASS", "Project root", str(PROJECT_ROOT))
    else:
        _result("FAIL", "Project root", "pyproject.toml not found")
        failures += 1

    _result("PASS", "Python", sys.version.split()[0])
    in_venv = sys.prefix != sys.base_prefix
    _result("PASS" if in_venv else "WARN", "Virtual environment", str(in_venv))

    packages = {
        "pytest": "pytest",
        "OpenAI SDK": "openai",
        "Pydantic": "pydantic",
        "python-dotenv": "dotenv",
        "FastAPI": "fastapi",
    }
    for label, module_name in packages.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            _result("PASS", label, str(version))
        except ImportError:
            _result("FAIL", label, "not installed")
            failures += 1

    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path, override=True)
    _result("PASS" if env_path.exists() else "WARN", ".env file", "present" if env_path.exists() else "missing")
    _result(
        "PASS" if os.getenv("OPENAI_API_KEY") else "WARN",
        "OPENAI_API_KEY",
        "configured" if os.getenv("OPENAI_API_KEY") else "not configured",
    )
    _result(
        "PASS" if os.getenv("OPENAI_VECTOR_STORE_ID") else "WARN",
        "OPENAI_VECTOR_STORE_ID",
        "configured" if os.getenv("OPENAI_VECTOR_STORE_ID") else "not configured",
    )
    smoke_enabled = os.getenv("RUN_OPENAI_SMOKE_TESTS") == "1"
    _result(
        "PASS" if smoke_enabled else "WARN",
        "OpenAI smoke opt-in",
        "enabled" if smoke_enabled else "disabled",
    )
    claim_model = bool(
        os.getenv("INSURANCE_CLAIM_EXTRACTOR_MODEL")
        or os.getenv("OPENAI_MODEL")
    )
    judgment_models = bool(
        os.getenv("INDEPENDENT_JUDGMENT_MODEL_A")
        and os.getenv("INDEPENDENT_JUDGMENT_MODEL_B")
    )
    _result(
        "PASS" if claim_model else "WARN",
        "Claim extraction model",
        "configured" if claim_model else "not configured",
    )
    _result(
        "PASS" if judgment_models else "WARN",
        "Two independent judgment models",
        "configured" if judgment_models else "not configured",
    )
    final_answer_model = bool(
        os.getenv("FINAL_ANSWER_MODEL") or os.getenv("OPENAI_MODEL")
    )
    _result(
        "PASS" if final_answer_model else "WARN",
        "Final answer model",
        "configured" if final_answer_model else "not configured",
    )

    git = _git_executable()
    if git is None:
        _result("WARN", "Git command", "not found in PATH or GitHub Desktop")
        return failures
    source = "PATH" if shutil.which("git") else "GitHub Desktop"
    _result("PASS", "Git command", f"{source} ({git})")
    root = _run_git(git, "rev-parse", "--show-toplevel")
    if root.returncode:
        _result("FAIL", "Git repository", "not detected")
        return failures + 1
    _result("PASS", "Git repository", root.stdout.strip())

    branch = _run_git(git, "branch", "--show-current")
    _result("PASS", "Current branch", branch.stdout.strip() or "detached")
    status = _run_git(git, "status", "--porcelain")
    _result(
        "PASS" if not status.stdout.strip() else "WARN",
        "Working tree",
        "clean" if not status.stdout.strip() else "has local changes",
    )
    ignored = _run_git(git, "check-ignore", "-q", ".env")
    _result(
        "PASS" if ignored.returncode == 0 else "FAIL",
        ".env ignored",
        str(ignored.returncode == 0),
    )
    if ignored.returncode != 0:
        failures += 1
    tracked = _run_git(git, "ls-files", "--error-unmatch", ".env")
    _result(
        "PASS" if tracked.returncode != 0 else "FAIL",
        ".env untracked",
        str(tracked.returncode != 0),
    )
    if tracked.returncode == 0:
        failures += 1
    return failures


if __name__ == "__main__":
    raise SystemExit(check_local_environment())
