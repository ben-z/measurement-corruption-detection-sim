#!/bin/bash

# This script is run after the project is created to auto-initialize dependencies

set -o errexit -o nounset -o pipefail -o xtrace

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export DEBIAN_FRONTEND=noninteractive

pushd ${SCRIPT_DIR}/backend
/opt/miniconda/bin/conda env create --file environment.yml
/opt/miniconda/bin/conda activate bsim-backend
pip install -r requirements.txt
popd

pushd ${SCRIPT_DIR}/frontend
npm ci
popd
