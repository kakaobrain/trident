#!/bin/bash

bash install_dependencies.sh
pip uninstall -y trident
pip install .
