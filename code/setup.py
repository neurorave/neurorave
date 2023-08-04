from setuptools import setup, find_packages

setup(
    name="raving_fader",
    version="1.0.0",
    description="Learning expressive control on RAVE for deep audio synthesis using Fader Networks with continuous audio descriptors attributes",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    url="https://forge-2.ircam.fr/acids/team/collaboration/raving-fader.git",
    author="ACIDS",
    license="UNLICENSED",
    packages=find_packages(),
    zip_safe=False,
)
