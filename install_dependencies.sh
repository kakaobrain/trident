#!/bin/bash

if [ ! -d ".triton" ]; then
    git clone https://github.com/openai/triton.git .triton
fi

pushd .triton/python
git pull
git reset --hard 5df904233c11a65bd131ead7268f84cca7804275
pip3 uninstall -y triton
pip3 install .
popd
