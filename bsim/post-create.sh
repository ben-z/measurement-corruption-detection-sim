#!/bin/bash

# This script is run after the project is created to auto-initialize dependencies

set -o errexit -o nounset -o pipefail -o xtrace

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export DEBIAN_FRONTEND=noninteractive

pushd ${SCRIPT_DIR}/backend
apt-get update && apt-get install -y gcc cmake gfortran libopenblas-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
popd

pushd ${SCRIPT_DIR}/frontend
npm ci
popd
