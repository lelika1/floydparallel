#!/usr/bin/env bash

LOW_SIZE=100
HIGH_SIZE=10000
TEST_COUNT=1000

ITER=0
# for SIZE in $(shuf -i "$LOW_SIZE-$HIGH_SIZE" -n $TEST_COUNT); do
for SIZE in 16 64 128 256 512 768 1024 2048 3072 4096 5120 6144 7168 8192; do
	echo "Iter $ITER: Size $SIZE"
	time ./gen "$SIZE" "inputs/input_$ITER.txt"
	ITER=$((ITER+1))
done
