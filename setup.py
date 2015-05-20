from setuptools import setup, find_packages

pandas_version = '0.16.0'

setup(
    name="IUVS",
    version="0.3",
    packages=find_packages(),

    install_requires=['astropy', 'pandas>='+pandas_version],

    scripts=['bin/show_versions'],

    # entry_points={
    #     "console_scripts": [
    #         'show_versions = iuvs.show_versions'
    #         ]
    # },

    # metadata
    author="K.-Michael Aye",
    author_email="michael.aye@lasp.colorado.edu",
    description="Software for handling MAVEN IUVS data",
    license="BSD 2-clause",
    keywords="MAVEN IUVS",
    url="http://lasp.colorado.edu",
)
