from setuptools import setup, find_packages

setup(
    name="skeletonactionrecognition",
    version="0.0.1",
    author="Kevin Schlegel",
    author_email="kevinschlegel@cantab.net",
    description="A collection of skeleton-based human action recognition "
    "methods using PyTorch",
    url="https://github.com/kschlegel/SkeletonActionRecognition",
    package_data={"skeletonactionrecognition": ["py.typed"]},
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[],
    zip_safe=False,
)
