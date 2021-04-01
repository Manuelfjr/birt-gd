import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="birt-sgd", # Replace with your own username
    version="1.0.4",
    author="Manuel Ferreira Junior",
    author_email="ferreira.jr.ufpb@gmail.com",
    description=" Evaluation of clustering methods using Beta^3-IRT with descending gradient",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Manuelfjr/birt-sgd",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
        "Source Code": "https://github.com/Manuelfjr/birt-sgd"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)