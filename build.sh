#!/usr/bin/env bash

set -eufx -o pipefail

g++ --std=c++11 floyd_sequential.cpp -o ./fseq
g++ -fopenmp --std=c++11  floyd_mp.cpp -o ./fmp
nvcc --std=c++11  floyd_cuda_simple.cu -o ./fcudas