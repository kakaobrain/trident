# Copyright 2023 ⓒ Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

setuptools.setup(
    name="trident",
    version="1.0.0",
    description="A performance library for machine learning applications",
    author="Kakao Brain Corp",
    packages=setuptools.find_packages(),
    install_requires=[
        "pytest",
        "matplotlib",
        "pandas",
        "triton@git+https://github.com/openai/triton.git@main#subdirectory=python",
    ],
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache License 2.0",
        "Programming Language :: Python :: 3.8",
    ],
)
