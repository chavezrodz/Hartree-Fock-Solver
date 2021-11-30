#!/bin/bash

python3 -m cProfile -o simple_hist_timing.dat ../runner_diagram.py

echo

python3 profiling.py >> profile_results.txt