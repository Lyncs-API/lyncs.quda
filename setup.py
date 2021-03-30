import sys
from pathlib import Path
import fileinput
from lyncs_setuptools import setup, CMakeExtension, find_package

requirements = [
    "cupy",
    "lyncs-cppyy",
]

QUDA_CMAKE_ARGS = {
    "CMAKE_BUILD_TYPE": "RELEASE",
    "QUDA_BUILD_SHAREDLIB": "ON",
    "QUDA_BUILD_ALL_TESTS": "OFF",
    "QUDA_GPU_ARCH": "sm_60",
    "QUDA_FORCE_GAUGE": "ON",
    "QUDA_MPI": "OFF",
}


findMPI = find_package("MPI")
if findMPI["cxx_found"]:
    requirements.append("lyncs_mpi")
    QUDA_CMAKE_ARGS["QUDA_MPI"] = "ON"


def patch_include(builder, ext):
    'Replaces #include instances in header files that use <> with "" for relative includes'
    install_dir = builder.get_install_dir(ext) + "/include"
    for path in Path(install_dir).rglob("*.h"):
        with fileinput.FileInput(str(path), inplace=True, backup=".bak") as fp:
            for fline in fp:
                line = str(fline)
                if line.strip().startswith("#include"):
                    include = line.split()[1]
                    if include[0] == "<" and include[-1] == ">":
                        include = include[1:-1]
                        if (path.parents[0] / include).exists():
                            print(line.replace(f"<{include}>", f'"{include}"'), end="")
                            continue
                print(line, end="")


QUDA_CMAKE_ARGS = [key + "=" + val for key, val in QUDA_CMAKE_ARGS.items()]
print("QUDA options:\n", "\n".join(QUDA_CMAKE_ARGS))

setup(
    "lyncs_quda",
    exclude=["*.config"],
    ext_modules=[
        CMakeExtension(
            "lyncs_quda.lib",
            ".",
            ["-DQUDA_CMAKE_ARGS='-D%s'" % ";-D".join(QUDA_CMAKE_ARGS)],
            post_build=patch_include,
        )
    ],
    data_files=[(".", ["config.py.in"])],
    install_requires=requirements,
    keywords=["Lyncs", "quda", "Lattice QCD", "python", "interface",],
)
