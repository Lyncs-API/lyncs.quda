## A Python interface to QUDA for Lyncs

[![python](https://img.shields.io/pypi/pyversions/lyncs_quda.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_quda/)
[//]: #[![pypi](https://img.shields.io/pypi/v/lyncs_quda.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_quda/)
[![license](https://img.shields.io/github/license/Lyncs-API/lyncs.quda?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.quda/blob/master/LICENSE)
[![build & test](https://img.shields.io/github/workflow/status/Lyncs-API/lyncs.quda/build%20&%20test?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.quda/actions)
[//]: #[![codecov](https://img.shields.io/codecov/c/github/Lyncs-API/lyncs.quda?logo=codecov&logoColor=white)](https://codecov.io/gh/Lyncs-API/lyncs.quda)
[![pylint](https://img.shields.io/badge/pylint%20score-9.4%2F10-green?logo=python&logoColor=white)](http://pylint.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=codefactor&logoColor=white)](https://github.com/ambv/black)


[QUDA](http://lattice.github.io/quda/) is a library for performing calculations in lattice QCD on GPUs.


### Installation

This package at the moment is not distrubuted via pip!
(Mostly because we do not have a CI/CD for GPUs)

For installing the packages from source code, follow the following steps:

```bash
# Set the following variables appropriately
export QUDA_GPU_ARCH=sm_60

# Clone and install
git clone https://github.com/Lyncs-API/lyncs.quda
cd lyncs.quda
python setup.py develop
```
