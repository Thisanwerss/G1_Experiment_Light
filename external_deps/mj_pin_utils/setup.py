from setuptools import setup, find_packages

setup(
    name="mj_pin",  # Package name
    version="0.1.0",      # Package version
    author="",
    author_email="",
    description="Helper library for project using both MuJoCo and Pinocchio.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # Use markdown for README
    url="https://github.com/Atarilab/mj_pin_utils",  # Repository URL
    packages=find_packages(where=""),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-clause",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',   # Minimum Python version
    install_requires=[
        "mujoco",  # Example dependencies
        "pin",
        "robot_descriptions",
        "opencv-python",
    ],
)
