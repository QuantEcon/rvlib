from distutils.core import setup
import os

rootdir = os.path.abspath(os.path.dirname(__file__))
long_desc = open(os.path.join(rootdir, 'README.md')).read()

# Write a versions.py file for class attribute
VERSION = "0.0.1"

def write_version_py(filename=None):
    doc = ("\"\"\"\n" +
           "This is a VERSION file and should NOT be manually altered"
           + "\n\"\"\"")
    doc += "\nversion = \"%s\"" % VERSION

    if not filename:
        filename = os.path.join(
                os.path.dirname(__file__), "distributions", "version.py")

    f = open(filename, "w")
    try:
        f.write(doc)
    finally:
        f.close()

write_version_py()

# Setup

setup(name="Distributions.py",
      packages=["distributions"],
      version=VERSION,
      description="Probability distributions mimicking Distrbutions.jl",
      author="Daniel Csaba, Spencer Lyon",
      author_email="daniel.csaba@nyu.edu, spencer.lyon@stern.nyu.edu",
      url="https://github.com/QuantEcon/Distributions.py", # URL to the github repo
      keywords=["statistics", "distributions"],
      long_description=long_desc)
