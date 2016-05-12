#!/usr/bin/env bash

set -eufx -o pipefail

./gen $1 input
./fs input output
./fseq input output_seq
diff output output_seq
