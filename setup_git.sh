#!/usr/bin/env bash
set -e

git init
git add README.md requirements.txt setup_git.sh train.py src
git commit -m "Stabilize full quantum actor-critic (lr 5e-4, entropy reg, shaped reward)"
