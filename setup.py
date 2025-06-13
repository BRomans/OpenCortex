from setuptools import setup, find_packages

setup(
    name="OpenCortex",
    version="0.1.7",
    author="Michele Romani",
    author_email="michele.romani.gzl0@gmail.com",
    description="Software to stream EEG data, perform preprocessing, and train machine learning models to build real-time BCI applications.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/BRomans/OpenCortex",
    include_package_data=True,
    packages=find_packages(exclude=["data", "images", "notebooks", "tests", "tools", "export", "examples"]),
    entry_points={
        "console_scripts": [
            "opencortex=opencortex.__main__:run_cli",
        ],
    },
    python_requires=">=3.8",
)
