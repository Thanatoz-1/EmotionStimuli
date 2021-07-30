__author__ = "Tushar Dhyani"

import setuptools

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emotion",  # thanatoz-1, max-milian
    version="0.0.1",
    author="Tushar D, Maximilian W.",
    author_email="st176870@stud.uni-stuttgart.de, @stud.uni-stuttgart.de",
    description="Emotion role labelling project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Thanatoz-1/EmotionStimuli",
    project_urls={
        "Bug Tracker": "https://github.com/Thanatoz-1/EmotionStimuli/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
