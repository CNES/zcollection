# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""This script is the entry point for building, distributing and installing
this module using distutils/setuptools."""

import datetime
import distutils.command.sdist
import os
import pathlib
import re
import subprocess
from typing import List, Optional, Pattern, Tuple

import setuptools

# Working directory
WORKING_DIRECTORY = pathlib.Path(__file__).parent.absolute()


def update_version(path: pathlib.Path, pattern: Pattern[str],
                   version: str) -> None:
    """Update version in file at path"""
    with open(path, mode="r", encoding="utf-8") as stream:
        lines = stream.readlines()
    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is not None:
            lines[idx] = version
    with open(path, mode="w", encoding="utf-8") as stream:
        stream.write("".join(lines))


def execute(cmd: str) -> str:
    """Executes a command and returns the lines displayed on the standard
    output"""
    with subprocess.Popen(cmd,
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as process:
        stdout = process.stdout
        assert stdout is not None, "Unable to read stdout from command"
        return stdout.read().decode()


def get_version(
        module: pathlib.Path) -> Tuple[str, Optional[datetime.datetime]]:
    """Reads the version number"""

    # If the ".git" directory exists, this function is executed in the
    # development environment, otherwise it's a release.
    if not pathlib.Path(WORKING_DIRECTORY, ".git").exists():
        pattern = re.compile(r'return "(\d+\.\d+\.\d+)"')
        with open(module, mode="r", encoding="utf-8") as stream:
            for line in stream:
                match = pattern.search(line)
                if match:
                    return match.group(1), None
        raise AssertionError()

    stdout = execute("git describe --tags --dirty --long --always").strip()
    pattern = re.compile(r"([\w\d.]+)-(\d+)-g([\w\d]+)(?:-(dirty))?")
    match = pattern.search(stdout)
    if match is None:
        # No tag found, use the last commit
        pattern = re.compile(r"([\w\d]+)(?:-(dirty))?")
        match = pattern.search(stdout)
        assert match is not None, f"Unable to parse git output {stdout!r}"
        version = "0.0"
        sha1 = match.group(1)
    else:
        version = match.group(1)
        commits = int(match.group(2))
        sha1 = match.group(3)
        if commits != 0:
            version += f".dev{commits}"

    stdout = execute(f"git log  {sha1} -1 --format=\"%H %at\"")
    stdout = stdout.strip().split()
    date = datetime.datetime.utcfromtimestamp(int(stdout[1]))

    return version, date


def revision() -> str:
    """Returns the software version"""
    os.chdir(WORKING_DIRECTORY)
    module = pathlib.Path(WORKING_DIRECTORY, "zcollection", "version.py")

    version, date = get_version(module)
    if date is None:
        # The date is read in the development environment. If it is not defined,
        # the files describing the version of the library should not be updated.
        return version

    # Conda/SonarCube configuration files are not present in the distribution,
    # but only in the GIT repository of the source code.
    meta = pathlib.Path(WORKING_DIRECTORY, "conda", "meta.yaml")
    if meta.exists():
        update_version(meta, re.compile(r'% set version = ".*" %}'),
                       f'{{% set version = "{version}" %}}\n')

    sonar_cube = pathlib.Path(WORKING_DIRECTORY, "sonar-project.properties")
    if sonar_cube.exists():
        update_version(sonar_cube,
                       re.compile(r"sonar.projectVersion=\d+\.\d+\.\d+"),
                       f"sonar.projectVersion={version}\n")

    # Updating the version number description for sphinx
    conf = pathlib.Path(WORKING_DIRECTORY, "docs", "source", "conf.py")
    with open(conf, mode="r", encoding="utf-8") as stream:
        lines = stream.readlines()
    pattern = re.compile(r"(\w+)\s+=\s+(.*)")

    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is not None:
            if match.group(1) == "version":
                lines[idx] = f"version = {version!r}\n"
            elif match.group(1) == "release":
                lines[idx] = f"release = {version!r}\n"
            elif match.group(1) == "copyright":
                lines[idx] = f"copyright = '({date.year}, CNES/CLS)'\n"

    with open(conf, "w", encoding="utf-8") as stream:
        stream.write("".join(lines))

    # Finally, write the file containing the version number.
    with open(module, "w", encoding="utf-8") as handler:
        handler.write(f'''"""
Get software version information
================================
"""


def release() -> str:
    """Returns the software version number"""
    return "{version}"


def date() -> str:
    """Returns the creation date of this release"""
    return "{date.strftime('%d %B %Y')}"
''')
    return version


def find_namespace_packages(path: pathlib.Path) -> List[str]:
    """Find all namespace packages in path"""
    result = []
    for item in pathlib.Path(path).glob("**/__init__.py"):
        result.append(str(item.parent.relative_to(path.parent)))
    return result


REQUIRES = [
    "dask",
    "distributed",
    "fsspec",
    "numcodecs",
    "numpy>=1.20",
    "pandas",
    "xarray",
    "zarr",
]


class SDist(distutils.command.sdist.sdist):
    """Custom sdist command that copies the pytest configuration file
    into the package"""
    user_options = distutils.command.sdist.sdist.user_options

    def run(self):
        """A command's raison d'etre: carry out the action"""
        source = WORKING_DIRECTORY.joinpath("conftest.py")
        target = WORKING_DIRECTORY.joinpath("zcollection", "conftest.py")
        source.rename(target)
        try:
            super().run()
        finally:
            target.rename(source)


def long_description():
    """Reads the README file"""
    with open(pathlib.Path(WORKING_DIRECTORY, "README.rst")) as stream:
        return stream.read()


setuptools.setup(
    author="CNES/CLS",
    author_email="fbriol@gmail.com",
    cmdclass={
        "sdist": SDist,
    },  # type: ignore
    description="Zarr Collection",
    install_requires=REQUIRES,
    license="BSD License",
    long_description=long_description(),
    long_description_content_type='text/x-rst',
    name="zcollection",
    package_data={"": ["*.json"]},
    packages=find_namespace_packages(pathlib.Path("zcollection")),
    python_requires=">=3.8",
    tests_require=REQUIRES + ["pytest"],
    url="https://github.com/CNES/zcollection",
    version=revision(),
)
