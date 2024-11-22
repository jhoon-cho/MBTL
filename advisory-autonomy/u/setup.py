import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="u", # Replace with your own username
    version="0.0.1",
    author="Zee Yan",
    author_email="zeexyan@gmail.com",
    description="Utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhongxiayan/util",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=[os.path.join('scripts', x) for x in os.listdir('scripts')]
)
