from setuptools import setup, find_packages
from embedKB import __version__

setup(
    name='embedKB',
    version=__version__,
    description='Embedding models for knowledge bases expressed in a general framework.',
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/zafarali/embedKB',
    author='Zafarali Ahmed and Charles C Onu',
    author_email='zafarali.ahmed@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
    python_requires='>=3.4.3',
    # entry_points={
    #     'console_scripts':[
    #         'embedKB=embedKB:data'
    #     ]
    # }
)