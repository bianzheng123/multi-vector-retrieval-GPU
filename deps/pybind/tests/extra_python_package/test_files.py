import contextlib
import os
import string
import subprocess
import sys
import tarfile
import zipfile

# These tests must be run explicitly
# They require CMake 3.15+ (--install)

DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(os.path.dirname(DIR))

PKGCONFIG = """\
prefix=${{pcfiledir}}/../../
includedir=${{prefix}}/include

Name: pybind
Description: Seamless operability between C++11 and Python
Version: {VERSION}
Cflags: -I${{includedir}}
"""


main_headers = {
    "include/pybind/attr.h",
    "include/pybind/buffer_info.h",
    "include/pybind/cast.h",
    "include/pybind/chrono.h",
    "include/pybind/common.h",
    "include/pybind/complex.h",
    "include/pybind/eigen.h",
    "include/pybind/embed.h",
    "include/pybind/evaluation.h",
    "include/pybind/functional.h",
    "include/pybind/gil.h",
    "include/pybind/iostream.h",
    "include/pybind/numpy.h",
    "include/pybind/operators.h",
    "include/pybind/options.h",
    "include/pybind/pybind.h",
    "include/pybind/pytypes.h",
    "include/pybind/stl.h",
    "include/pybind/stl_bind.h",
}

detail_headers = {
    "include/pybind/detail/class.h",
    "include/pybind/detail/common.h",
    "include/pybind/detail/descr.h",
    "include/pybind/detail/init.h",
    "include/pybind/detail/internals.h",
    "include/pybind/detail/type_caster_base.h",
    "include/pybind/detail/typeid.h",
}

eigen_headers = {
    "include/pybind/eigen/matrix.h",
    "include/pybind/eigen/tensor.h",
}

stl_headers = {
    "include/pybind/stl/filesystem.h",
}

cmake_files = {
    "share/cmake/pybind/FindPythonLibsNew.cmake",
    "share/cmake/pybind/pybind11Common.cmake",
    "share/cmake/pybind/pybind11Config.cmake",
    "share/cmake/pybind/pybind11ConfigVersion.cmake",
    "share/cmake/pybind/pybind11NewTools.cmake",
    "share/cmake/pybind/pybind11Targets.cmake",
    "share/cmake/pybind/pybind11Tools.cmake",
}

pkgconfig_files = {
    "share/pkgconfig/pybind.pc",
}

py_files = {
    "__init__.py",
    "__main__.py",
    "_version.py",
    "commands.py",
    "py.typed",
    "setup_helpers.py",
}

headers = main_headers | detail_headers | eigen_headers | stl_headers
src_files = headers | cmake_files | pkgconfig_files
all_files = src_files | py_files


sdist_files = {
    "pybind",
    "pybind/include",
    "pybind/include/pybind",
    "pybind/include/pybind/detail",
    "pybind/include/pybind/eigen",
    "pybind/include/pybind/stl",
    "pybind/share",
    "pybind/share/cmake",
    "pybind/share/cmake/pybind",
    "pybind/share/pkgconfig",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "LICENSE",
    "MANIFEST.in",
    "README.rst",
    "PKG-INFO",
}

local_sdist_files = {
    ".egg-info",
    ".egg-info/PKG-INFO",
    ".egg-info/SOURCES.txt",
    ".egg-info/dependency_links.txt",
    ".egg-info/not-zip-safe",
    ".egg-info/top_level.txt",
}


def read_tz_file(tar: tarfile.TarFile, name: str) -> bytes:
    start = tar.getnames()[0] + "/"
    inner_file = tar.extractfile(tar.getmember(f"{start}{name}"))
    assert inner_file
    with contextlib.closing(inner_file) as f:
        return f.read()


def normalize_line_endings(value: bytes) -> bytes:
    return value.replace(os.linesep.encode("utf-8"), b"\n")


def test_build_sdist(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)

    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", f"--outdir={tmpdir}"], check=True
    )

    (sdist,) = tmpdir.visit("*.tar.gz")

    with tarfile.open(str(sdist), "r:gz") as tar:
        start = tar.getnames()[0] + "/"
        version = start[9:-1]
        simpler = {n.split("/", 1)[-1] for n in tar.getnames()[1:]}

        setup_py = read_tz_file(tar, "setup.py")
        pyproject_toml = read_tz_file(tar, "pyproject.toml")
        pkgconfig = read_tz_file(tar, "pybind/share/pkgconfig/pybind.pc")
        cmake_cfg = read_tz_file(
            tar, "pybind/share/cmake/pybind/pybind11Config.cmake"
        )

    assert (
        'set(pybind11_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")'
        in cmake_cfg.decode("utf-8")
    )

    files = {f"pybind/{n}" for n in all_files}
    files |= sdist_files
    files |= {f"pybind{n}" for n in local_sdist_files}
    files.add("pybind.egg-info/entry_points.txt")
    files.add("pybind.egg-info/requires.txt")
    assert simpler == files

    with open(os.path.join(MAIN_DIR, "tools", "setup_main.py.in"), "rb") as f:
        contents = (
            string.Template(f.read().decode("utf-8"))
            .substitute(version=version, extra_cmd="")
            .encode("utf-8")
        )
    assert setup_py == contents

    with open(os.path.join(MAIN_DIR, "tools", "pyproject.toml"), "rb") as f:
        contents = f.read()
    assert pyproject_toml == contents

    simple_version = ".".join(version.split(".")[:3])
    pkgconfig_expected = PKGCONFIG.format(VERSION=simple_version).encode("utf-8")
    assert normalize_line_endings(pkgconfig) == pkgconfig_expected


def test_build_global_dist(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)
    monkeypatch.setenv("PYBIND11_GLOBAL_SDIST", "1")
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmpdir)], check=True
    )

    (sdist,) = tmpdir.visit("*.tar.gz")

    with tarfile.open(str(sdist), "r:gz") as tar:
        start = tar.getnames()[0] + "/"
        version = start[16:-1]
        simpler = {n.split("/", 1)[-1] for n in tar.getnames()[1:]}

        setup_py = read_tz_file(tar, "setup.py")
        pyproject_toml = read_tz_file(tar, "pyproject.toml")
        pkgconfig = read_tz_file(tar, "pybind/share/pkgconfig/pybind.pc")
        cmake_cfg = read_tz_file(
            tar, "pybind/share/cmake/pybind/pybind11Config.cmake"
        )

    assert (
        'set(pybind11_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")'
        in cmake_cfg.decode("utf-8")
    )

    files = {f"pybind/{n}" for n in all_files}
    files |= sdist_files
    files |= {f"pybind11_global{n}" for n in local_sdist_files}
    assert simpler == files

    with open(os.path.join(MAIN_DIR, "tools", "setup_global.py.in"), "rb") as f:
        contents = (
            string.Template(f.read().decode())
            .substitute(version=version, extra_cmd="")
            .encode("utf-8")
        )
        assert setup_py == contents

    with open(os.path.join(MAIN_DIR, "tools", "pyproject.toml"), "rb") as f:
        contents = f.read()
        assert pyproject_toml == contents

    simple_version = ".".join(version.split(".")[:3])
    pkgconfig_expected = PKGCONFIG.format(VERSION=simple_version).encode("utf-8")
    assert normalize_line_endings(pkgconfig) == pkgconfig_expected


def tests_build_wheel(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "-w", str(tmpdir)], check=True
    )

    (wheel,) = tmpdir.visit("*.whl")

    files = {f"pybind/{n}" for n in all_files}
    files |= {
        "dist-info/LICENSE",
        "dist-info/METADATA",
        "dist-info/RECORD",
        "dist-info/WHEEL",
        "dist-info/entry_points.txt",
        "dist-info/top_level.txt",
    }

    with zipfile.ZipFile(str(wheel)) as z:
        names = z.namelist()

    trimmed = {n for n in names if "dist-info" not in n}
    trimmed |= {f"dist-info/{n.split('/', 1)[-1]}" for n in names if "dist-info" in n}
    assert files == trimmed


def tests_build_global_wheel(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)
    monkeypatch.setenv("PYBIND11_GLOBAL_SDIST", "1")

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "-w", str(tmpdir)], check=True
    )

    (wheel,) = tmpdir.visit("*.whl")

    files = {f"data/data/{n}" for n in src_files}
    files |= {f"data/headers/{n[8:]}" for n in headers}
    files |= {
        "dist-info/LICENSE",
        "dist-info/METADATA",
        "dist-info/WHEEL",
        "dist-info/top_level.txt",
        "dist-info/RECORD",
    }

    with zipfile.ZipFile(str(wheel)) as z:
        names = z.namelist()

    beginning = names[0].split("/", 1)[0].rsplit(".", 1)[0]
    trimmed = {n[len(beginning) + 1 :] for n in names}

    assert files == trimmed
