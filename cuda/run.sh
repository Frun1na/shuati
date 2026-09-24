#!/usr/bin/env bash
# 用法: ./run.sh ex01          运行你的练习
#       ./run.sh ex01 --sol    运行参考答案
set -e
cd "$(dirname "$0")"
dir=exercises; [[ "$2" == "--sol" ]] && dir=solutions
f=$(ls $dir/$1_*.py | head -1)
PYTHONPATH=. NUMBA_ENABLE_CUDASIM=1 python3 "$f"
