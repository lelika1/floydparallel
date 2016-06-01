#!/usr/bin/env bash

# FILES=./inputs/*

# echo "cuda_req"
# echo "v time" >> results/blocked_req_time.txt
# for f in $FILES
# do
#   echo "$f"
#   ./freg "$f" "output" >> results/blocked_req_time.txt
#   ./freg "$f" "output" >> results/blocked_req_time.txt
#   ./freg "$f" "output" >> results/blocked_req_time.txt
# done

# echo "cuda_blocked"
# echo "v time" >> results/blocked_time.txt
# for f in $FILES
# do
#   echo "$f"
#   ./fblocked "$f" "output" >> results/blocked_time.txt
#   ./fblocked "$f" "output" >> results/blocked_time.txt
#   ./fblocked "$f" "output" >> results/blocked_time.txt
# done

# echo "cuda"
# echo "v time" >> results/blocked_time.txt
# for f in $FILES
# do
#   echo "$f"
#   ./fcudas "$f" "output" >> results/cuda_time.txt
#   ./fcudas "$f" "output" >> results/cuda_time.txt
#   ./fcudas "$f" "output" >> results/cuda_time.txt
# done

# echo "open_mp"
# echo "v time" >> results/blocked_time.txt
# for f in $FILES
# do
#   echo "$f"
#   ./fmp "$f" "output_mp" >> results/mp_time.txt
#   ./fmp "$f" "output_mp" >> results/mp_time.txt
#   ./fmp "$f" "output_mp" >> results/mp_time.txt
# done

# echo "seq"
# echo "v time" >> results/blocked_time.txt
# for f in $FILES
# do
#   echo "$f"
#   ./fseq "$f" "output_s" >> results/seq_time.txt
#   ./fseq "$f" "output_s" >> results/seq_time.txt
#   ./fseq "$f" "output_s" >> results/seq_time.txt
# done