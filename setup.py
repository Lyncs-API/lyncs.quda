import sys
import os
from lyncs_setuptools import setup, CMakeExtension, find_package

requirements = [
    "lyncs-cppyy",
    "lyncs-utils>=0.2.2",
    "appdirs",
]

findCUDA = find_package("CUDA")
if findCUDA["found"]:
    requirements.append(f"cupy-cuda{findCUDA['version'].replace('.','')}")

print("LYNCS_QUDA requirements:", *requirements, sep="\n")


QUDA_CMAKE_ARGS = {
    "CMAKE_BUILD_TYPE": "RELEASE",
    "QUDA_BUILD_SHAREDLIB": "ON",
    "QUDA_BUILD_ALL_TESTS": "OFF",
    "QUDA_GPU_ARCH": os.environ.get("QUDA_GPU_ARCH", "sm_60"),
    "QUDA_MPI": os.environ.get("QUDA_MPI", "OFF"),
    "QUDA_MULTIGRID": "ON",
    "QUDA_CLOVER_DYNAMIC": "OFF",
    "QUDA_CLOVER_RECONSTRUCT": "OFF",
}


findMPI = find_package("MPI")
if findMPI["cxx_found"] and os.environ.get("QUDA_MPI", None) in (None, "ON"):
    requirements.append("lyncs_mpi")
    QUDA_CMAKE_ARGS["QUDA_MPI"] = "ON"

QUDA_CMAKE_ARGS = [key + "=" + val for key, val in QUDA_CMAKE_ARGS.items()]
print("QUDA options:", *QUDA_CMAKE_ARGS, sep="\n")


def post_build(*args, **kwargs):
    exec(open("post_build.py").read(), globals())
    post_build(*args, **kwargs)


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
    data_files=[(".", ["post_build.py"])],
    install_requires=requirements,
    extras_require={
        "hmc": ["aim"],
    },
    keywords=[
        "Lyncs",
        "quda",
        "Lattice QCD",
        "python",
        "interface",
    ],
)
