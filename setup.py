from setuptools import setup, find_packages

setup(
    name="lngraph",
    version="0.1.0",
    description="Data-science platform for analyzing the public Lightning Network channel graph",
    author="clankwright",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
