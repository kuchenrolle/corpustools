#!/usr/bin/python3
from glob import glob
from os import system
from os.path import basename
from os.path import splitext

from setuptools import Command
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup


# taken from http://stackoverflow.com/a/3780822
class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./src/*.egg-info')


tst = Extension("corpustools.tst",
                sources = ["src/corpustools/tst.pyx"])

setup(
    name='corpustools',
    version='0.1.0',
    license='MIT',
    description='Collection of tools for working with text data and corpora.',
    long_description=open('README.md').read(),
    author='Christian Adam',
    author_email='kuchenrolle@googlemail.com',
    url='https://github.com/kuchenrolle/corpustools',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    ext_modules=[tst],
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        # 'Programming Language :: Python',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        # 'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
    ],
    keywords=[
        'corpus', 'text mining', 'ternary search tree', 'ngrams'
    ],
    install_requires=[
        'spacy', 'pandas', 'psutil', 'pyndl', "cython"
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        # 'console_scripts': [
        #     'corpustools = corpustools.__main__:main',
        # ]
    },
    cmdclass={
        'clean': CleanCommand,
    },
)
