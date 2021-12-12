import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="birt-gd", # Replace with your own username
    version="0.1.19",
    author="Manuel Ferreira Junior",
    author_email="ferreira.jr.ufpb@gmail.com",
    description="BIRTGD is an implementation of Beta3-irt using gradient descent.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Manuelfjr/birt-gd",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
        "Source Code": "https://github.com/Manuelfjr/birt-gd/blob/main/src/birt/__init__.py"
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
        'tqdm',
        'seaborn',
        'matplotlib',
        'scikit-learn'
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude="mc_analysis"),
    python_requires=">=3.6",
)
