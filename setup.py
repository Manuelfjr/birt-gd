import setuptools

# All package metadata (version, dependencies, python_requires, classifiers)
# lives in setup.cfg. Do not pass kwargs here — explicit setup() args would
# silently override setup.cfg instead of raising a conflict.
setuptools.setup()
