#!/usr/bin/env bash

set -eufx -o pipefail

g++ --std=c++11 graph_generator.cpp -o ./gen
g++ --std=c++11 floyd_sequential.cpp -o ./fseq
g++ -fopenmp --std=c++11 floyd_mp.cpp -o ./fmp
nvcc --std=c++11 floyd_cuda_simple.cu -o ./fcudas
nvcc --std=c++11 floyd_cuda_blocked.cu -o ./fblocked
nvcc --std=c++11 floyd_cuda_blocked_reg.cu -o ./freg
nvcc --std=c++11 floyd_blocked_staged.cu -o ./fstaged
nvcc --std=c++11 floyd_modified.cu -o ./fmodified
