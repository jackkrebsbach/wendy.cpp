#!/bin/bash
set -e  # Exit on error
cmake -S . -B build
cmake --build build
./build/example
