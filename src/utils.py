# src/utils.py
import subprocess


def get_git_commit_hash() -> str:
    """Gets the current git commit hash."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
