import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lmdec",
    version="0.0.2",
    author="Tim Nonet",
    author_email="tnonet@mit.edu",
    description="A simple to install Library for Large matrix SVD and PCA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TNonet/lmdec",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['dask>=2.9.0',
                      'numpy>=1.17.0',
                      'pandas_plink>=2.0.0',
                      'numba>=0.45.1',
                      'scipy>=1.3.1',
                      'zarr>=2.3.2',
                      'numcodecs>=0.6.3',
                      'h5py>=2.9.0',
                      ]
)