#!/usr/bin/env bash

FILES=./inputs/*

# echo "cuda"
# for f in $FILES
# do
#   echo "$f"
#   ./fblocked "$f" "output" >> results/blocked_time.txt
#   ./fblocked "$f" "output" >> results/blocked_time.txt
#   ./fblocked "$f" "output" >> results/blocked_time.txt
# done

# echo "cuda"
# for f in $FILES
# do
#   echo "$f"
#   ./fcudas "$f" "output" >> results/cuda_time.txt
#   ./fcudas "$f" "output" >> results/cuda_time.txt
#   ./fcudas "$f" "output" >> results/cuda_time.txt
# done

# echo "open_mp"
# for f in $FILES
# do
#   echo "$f"
#   ./fmp "$f" "output_mp" >> results/mp_time.txt
#   ./fmp "$f" "output_mp" >> results/mp_time.txt
#   ./fmp "$f" "output_mp" >> results/mp_time.txt
# done

# echo "seq"
# for f in $FILES
# do
#   echo "$f"
#   ./fseq "$f" "output_s" >> results/seq_time.txt
#   ./fseq "$f" "output_s" >> results/seq_time.txt
#   ./fseq "$f" "output_s" >> results/seq_time.txt
# done