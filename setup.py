import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="birt-sgd", # Replace with your own username
    version="0.1.15",
    author="Manuel Ferreira Junior",
    author_email="ferreira.jr.ufpb@gmail.com",
    description="BIRTSGD is an implementation of Beta3-irt using gradient descent.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Manuelfjr/birt-sgd",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
        "Source Code": "https://github.com/Manuelfjr/birt-sgd/blob/main/src/birt/__init__.py"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'tensorflow',
        'pandas',
        'tqdm'
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude="mc_analysis"),
    python_requires=">=3.6",
)
