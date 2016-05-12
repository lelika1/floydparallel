#!/usr/bin/env bash

LOW_SIZE=100
HIGH_SIZE=10000
TEST_COUNT=1000

ITER=0
for SIZE in $(shuf -i "$LOW_SIZE-$HIGH_SIZE" -n $TEST_COUNT); do
	echo "Iter $ITER: Size $SIZE"
	time ./gen "$SIZE" "inputs/input_$ITER.txt"
	ITER=$((ITER+1))
done
