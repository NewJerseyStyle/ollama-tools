# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from ollama_tools import __version__

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ollama-tools',
    version=__version__,
    description='A workaround for models on Ollama that does not support tool calling',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Yuan XU',
    author_email='dev.source@outlook.com',
    url='https://github.com/NewJerseyStyle',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['ollama'],
    entry_points={
        'console_scripts': [
            'ollama-tools=ollama_tools.server:main',
        ],
    },
    extras_require={
        'full': ['fastapi', 'uvicorn'],
    },
)

