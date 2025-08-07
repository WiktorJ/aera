from setuptools import setup, find_packages

setup(
    name="autonomous",
    version="0.1.0",
    description="Autonomous robotics environments for reinforcement learning",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "mujoco",
        "numpy",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
