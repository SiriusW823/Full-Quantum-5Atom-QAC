#!/usr/bin/env bash
set -e

git init
git add README.md requirements.txt setup_git.sh train.py src
git commit -m "Add dynamic-depth quantum actor-critic with stabilized training"
