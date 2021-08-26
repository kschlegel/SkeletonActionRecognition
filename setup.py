from setuptools import setup, find_packages

setup(
    name="SkeletonActionRecognition",
    version="0.0.1",
    author="Kevin Schlegel",
    author_email="kevinschlegel@cantab.net",
    description="A collection of skeleton-based human action recognition "
    "methods using PyTorch",
    url="https://github.com/kschlegel/SkeletonActionRecognition",
    package_data={"shar": ["py.typed"]},
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['opencv-python>=4.5.2'],
    zip_safe=False,
)
