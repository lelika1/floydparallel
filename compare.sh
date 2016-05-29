#!/usr/bin/env bash

set -eufx -o pipefail

echo "$1"
./gen $1 input
./fcudas input output
./fblocked input output_blocked
diff output output_blocked
