import setuptools


setuptools.setup(
    name="roll-up-list",
    version="0.0.1",
    author="NovaVision AI",
    author_email="info@novavision.ai",
    description="Roll-Up List component for NovaVision",
    url="https://github.com/Duyguersoy/cap-roll-up-list",
    license="MIT",

    install_requires=[
        "numpy",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    packages=[
        "novavision.package",
        "novavision.package.executors",
        "novavision.package.models",
        "novavision.package.utils",
    ],

    package_dir={
        "novavision.package": "src",
    },

    python_requires=">=3.7",
)