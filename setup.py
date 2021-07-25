import sys
import os
from lyncs_setuptools import setup, CMakeExtension, find_package
from post_build import post_build

requirements = [
    "cupy",
    "lyncs-cppyy",
    "lyncs-utils>=0.2.2",
]

QUDA_CMAKE_ARGS = {
    "CMAKE_BUILD_TYPE": "RELEASE",
    "QUDA_BUILD_SHAREDLIB": "ON",
    "QUDA_BUILD_ALL_TESTS": "OFF",
    "QUDA_GPU_ARCH": os.environ.get("QUDA_GPU_ARCH", "sm_60"),
    "QUDA_FORCE_GAUGE": "ON",
    "QUDA_MPI": "OFF",
}


findMPI = find_package("MPI")
if findMPI["cxx_found"]:
    requirements.append("lyncs_mpi")
    QUDA_CMAKE_ARGS["QUDA_MPI"] = "ON"


QUDA_CMAKE_ARGS = [key + "=" + val for key, val in QUDA_CMAKE_ARGS.items()]
print("QUDA options:", *QUDA_CMAKE_ARGS, sep="\n")

setup(
    "lyncs_quda",
    exclude=["*.config"],
    ext_modules=[
        CMakeExtension(
            "lyncs_quda.lib",
            ".",
            ["-DQUDA_CMAKE_ARGS='-D%s'" % ";-D".join(QUDA_CMAKE_ARGS)],
            post_build=post_build,
        )
    ],
    data_files=[(".", ["config.py.in"])],
    install_requires=requirements,
    keywords=[
        "Lyncs",
        "quda",
        "Lattice QCD",
        "python",
        "interface",
    ],
)
