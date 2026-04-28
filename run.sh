#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
source venv/bin/activate 2>/dev/null || true
exec python main.py "$@"
