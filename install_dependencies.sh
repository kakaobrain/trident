#!/bin/bash

pip3 uninstall -y triton
git clone https://github.com/openai/triton.git
pushd triton/python
pip3 install .
popd
rm -rf triton