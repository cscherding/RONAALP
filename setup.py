import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RONAALP",
    version="0.0.1",
    author="",
    author_email="",
    description="RONAALP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cscherding/RONAALP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
