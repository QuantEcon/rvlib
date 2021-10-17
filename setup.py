from setuptools import setup
import os

rootdir = os.path.abspath(os.path.dirname(__file__))
with open("README.md", "r", encoding="utf8") as file:
    long_description = file.read()

# Write a versions.py file for class attribute
VERSION = "0.0.6"


def write_version_py(filename=None):
    doc = ("\"\"\"\n" +
           "This is a VERSION file and should NOT be manually altered" +
           "\n\"\"\"")
    doc += "\nversion = \"%s\"" % VERSION

    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), "rvlib", "version.py")

    f = open(filename, "w")
    try:
        f.write(doc)
    finally:
        f.close()


write_version_py()

# Setup
setup(name="rvlib",
      packages=["rvlib"],
      setup_requires=["cffi>=1.0.0","PyYAML"],
      scripts=["./build_interface.py"],
      cffi_modules=["build_lib.py:ffi"],
      install_requires=["cffi>=1.0.0", "numba>=0.49", "numpy", "PyYAML"],
      include_package_data=True,
      version=VERSION,
      description="Probability distributions mimicking Distrbutions.jl",
      author="Daniel Csaba, Spencer Lyon",
      author_email="daniel.csaba@nyu.edu, spencer.lyon@stern.nyu.edu",
      url="https://github.com/QuantEcon/rvlib", # URL to the github repo
      keywords=["statistics", "distributions"],
      long_description=long_description,
      long_description_content_type='text/markdown')
