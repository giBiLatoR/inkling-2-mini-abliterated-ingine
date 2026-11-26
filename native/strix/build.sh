#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
sdk=${ROCM_PATH:-/opt/rocm}
mkdir -p ../build-strix
output=$(mktemp ../build-strix/libingine_strix.XXXXXX.so)
trap 'rm -f "$output"' EXIT
"$sdk/bin/hipcc" -O3 --offload-arch=gfx1151 -fPIC -shared conv.hip gemv.hip experts.hip -o "$output"
chmod 755 "$output"
mv "$output" ../build-strix/libingine_strix.so
