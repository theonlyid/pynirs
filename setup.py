#!/usr/bin/env python

from distutils.core import setup

setup(
    name="pynirs",
    version="0.0.2",
    description="Package for analysis of nirs or IP timeseries data",
    author="Ali Zaidi",
    author_email="zaidi@icord.org",
    package_dir={"pynirs": "src/pynirs"},
    packages=["pynirs"],
)
