#!/bin/bash

if [ ! -d ".triton" ]; then
    git clone https://github.com/openai/triton.git .triton
fi

pushd .triton/python
git pull
git reset --hard 1465b573e8d8e4c707d579092001bcff0f1523ed
pip3 uninstall -y triton
pip3 install .
popd
