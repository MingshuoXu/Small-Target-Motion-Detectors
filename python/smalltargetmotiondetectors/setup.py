from setuptools import setup, find_packages

setup(
    name='smalltargetmotiondetectors',  
    version='2.4',  
    author='Mingshuo XU',  
    author_email='mingshuoxu@hotmail.com', 
    description='A package for detecting small targets in motion.',  
    long_description='A Python package for detecting small targets in motion. It provides algorithms and utilities for motion detection tasks.',  
    long_description_content_type='text/markdown',  
    url='https://github.com/MingshuoXu/Small-Target-Motion-Detectors/tree/main/python/smalltargetmotiondetectors',  
    packages=find_packages(), 
    keywords='motion-detection small-target bio-inspired-visual-model ',
    install_requires=[  
        'numpy',
        'scipy',
        'opencv-python',
        'matplotlib'
    ],
    classifiers=[  
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
    ],
)
