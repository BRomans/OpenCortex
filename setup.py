from setuptools import setup, find_packages

setup(
    name="UnicornPythonEssentialsToolkit",
    version="0.0.1",
    author="Michele Romani",
    author_email="michele.romani.gzl0@gmail.com",
    description="Package to analyze and manipulate data collected with Unicorn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/BRomans/UnicornPythonEssentialsToolkit",
    packages=find_packages(exclude=["data", "images", "notebooks"]),
    python_requires=">=3.8",
)
