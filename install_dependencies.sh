#!/bin/bash

if [ ! -d ".triton" ]; then
    git clone https://github.com/openai/triton.git .triton
fi

pushd .triton/python
git pull
pip3 uninstall -y triton
pip3 install -e .
popd