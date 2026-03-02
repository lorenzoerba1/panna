#!/usr/bin/env python

"""
USAGE: run-at.py COMMITS SCRIPT [SCRIPT_ARGS...]

This script accepts a (range of) git commits and the path to a python script and:

- Copies the script to the /tmp directory
- For each git commit in the range:
  - checks out the commit
  - runs the script stored in /tmp using `nix run .#python`
"""

import sys
import subprocess
import shutil
import tempfile
import os

def run_command(command, check=True):
    """Runs a command and checks for errors."""
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip())
        sys.exit(1)

    commits_range = sys.argv[1]
    script_path = sys.argv[2]
    additional_args = sys.argv[3:]

    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    # Get the list of commits
    rev_list_cmd = ["git", "rev-list", "--reverse", commits_range]
    result = run_command(rev_list_cmd)
    commits = result.stdout.strip().split()

    if not commits:
        print(f"No commits found for range: {commits_range}")
        sys.exit(0)

    print(f"Found {len(commits)} commits to process.")

    # Create a temporary file and copy the script content into it
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".py", dir="/tmp") as temp_script:
        temp_script_path = temp_script.name
        shutil.copy(script_path, temp_script_path)

    print(f"Copied script to {temp_script_path}")

    # Save the current git state to restore later
    original_branch_or_commit = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    if original_branch_or_commit == 'HEAD':
        # In detached HEAD state, get commit hash
        original_branch_or_commit = run_command(["git", "rev-parse", "HEAD"]).stdout.strip()

    try:
        for commit in commits:
            print(f"\n--- Processing commit {commit} ---")

            # Checkout the commit
            run_command(["git", "checkout", commit])

            # Run the script
            run_command(["nix", "run", ".#python", "--", temp_script_path] + additional_args)

    finally:
        print(f"\n--- Restoring original state: checking out {original_branch_or_commit} ---")
        # Restore the original state
        run_command(["git", "checkout", original_branch_or_commit])
        # Clean up the temporary script
        os.remove(temp_script_path)
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
