from setuptools import setup, find_packages

setup(
    name='smalltargetmotiondetectors',  
    version='2.2',  
    author='Mingshuo XU',  
    author_email='mingshuoxu@hotmail.com', 
    description='A package for detecting small targets in motion.',  
    long_description='A Python package for detecting small targets in motion. It provides algorithms and utilities for motion detection tasks.',  # 包的详细描述
    long_description_content_type='text/markdown',  # 描述的内容类型
    url='https://github.com/MingshuoXu/Small-Target-Motion-Detectors/tree/main/python/smalltargetmotiondetectors',  # 包的项目地址
    packages=find_packages(),  # 包含的包列表，find_packages()会自动查找当前目录下的所有包
    install_requires=[  # 依赖的其他包
        'numpy',
        'opencv-python',
        # 添加其他依赖项
    ],
    classifiers=[  # 分类器列表，用于指定该包适用于哪些Python版本、操作系统等
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
