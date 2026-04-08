#!/usr/bin/env python3
"""
Clean up experiment artifacts after a failed experiment run.

Usage:
    python cleanup_experiment.py           # Interactive mode
    python cleanup_experiment.py --dry-run # Preview what would be deleted
    python cleanup_experiment.py --force   # Delete without confirmation
"""

import argparse
import os
import shutil
from pathlib import Path


def get_experiment_artifacts(base_dir: Path):
    """Return list of (path, description) tuples for experiment artifacts."""
    artifacts = []

    # Only clean results/logs/ directory
    logs_dir = base_dir / "results" / "logs"
    if logs_dir.exists():
        for f in logs_dir.glob("*.log"):
            artifacts.append((f, "log file"))

    # Root-level gpu_manager.log
    gm_log = base_dir / "gpu_manager.log"
    if gm_log.exists():
        artifacts.append((gm_log, "gpu_manager log"))

    # results.tsv if exists
    tsv = base_dir / "results.tsv"
    if tsv.exists():
        artifacts.append((tsv, "results.tsv"))

    return artifacts


def cleanup(base_dir: Path, dry_run: bool = False, force: bool = False):
    artifacts = get_experiment_artifacts(base_dir)

    if not artifacts:
        print("No experiment artifacts found.")
        return

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Found {len(artifacts)} artifact(s) to clean:\n")
    for path, desc in artifacts:
        size = ""
        if path.exists() and path.is_file():
            size = f" ({path.stat().st_size / 1024 / 1024:.1f} MB)"
        elif path.exists() and path.is_dir():
            size = f" (dir)"
        print(f"  - {path.relative_to(base_dir)} [{desc}]{size}")

    if dry_run:
        print(f"\n[DRY RUN] Nothing deleted.")
        return

    if not force:
        response = input(f"\nDelete {len(artifacts)} artifact(s)? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    print("\nDeleting...")
    deleted = 0
    for path, _ in artifacts:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  Deleted: {path.relative_to(base_dir)}")
            deleted += 1
        except Exception as e:
            print(f"  Failed to delete {path}: {e}")

    print(f"\nDone. Deleted {deleted}/{len(artifacts)} item(s).")


def reset_git(base_dir: Path, dry_run: bool = False, force: bool = False):
    """Reset git to clean state."""
    print("\n" + "=" * 60)
    print("GIT RESET")
    print("=" * 60)

    import subprocess

    if dry_run:
        result = subprocess.run(["git", "status", "--short"], cwd=base_dir, capture_output=True, text=True)
        if result.stdout.strip():
            print("[DRY RUN] Git would reset to clean state.")
            print("\nModified/tracked files:")
            print(result.stdout)
        else:
            print("[DRY RUN] Git is already clean.")
        return

    if not force:
        response = input("Reset git to clean state (discard all uncommitted changes)? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Discard all uncommitted changes
    subprocess.run(["git", "checkout", "."], cwd=base_dir)
    # Remove untracked files (but not .claude/)
    result = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=base_dir, capture_output=True, text=True)
    for line in result.stdout.strip().split("\n"):
        if line.startswith("??"):
            f = line[3:].strip()
            if f == ".claude" or f.startswith(".claude/"):
                continue  # Keep .claude directory
            path = base_dir / f
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"  Removed: {f}")

    print("\nGit reset complete.")


def main():
    parser = argparse.ArgumentParser(description="Clean up experiment artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be deleted")
    parser.add_argument("--force", action="store_true", help="Delete without confirmation")
    parser.add_argument("--git-only", action="store_true", help="Only reset git, skip file cleanup")
    parser.add_argument("--files-only", action="store_true", help="Only clean files, skip git reset")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.resolve()

    print("=" * 60)
    print("EXPERIMENT CLEANUP")
    print("=" * 60)

    if not args.git_only:
        cleanup(base_dir, dry_run=args.dry_run, force=args.force)

    if not args.files_only:
        reset_git(base_dir, dry_run=args.dry_run, force=args.force)

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")


if __name__ == "__main__":
    main()
