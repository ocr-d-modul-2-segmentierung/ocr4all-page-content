from setuptools import setup, find_packages

setup(
    name='page_content_detection',
    version='0.0.1',
    packages=find_packages(),
    license='LGPL-v3.0',
    long_description=open("README.md").read(),
    include_package_data=True,
    author="Alexander Hartelt",
    author_email="alexander.hartelt@informatik.uni-wuerzburg.de",
    url="https://gitlab2.informatik.uni-wuerzburg.de/alh75dg/page-content.git",
    download_url='https://gitlab2.informatik.uni-wuerzburg.de/alh75dg/page-content.git',
    entry_points={
        'console_scripts': [
            'page_content_detect=pagecontent.scripts.scripts:main',
        ],
    },
    install_requires=open("requirements.txt").read().split(),
    extras_require={
        'tf_cpu': ['tensorflow>=1.6.0'],
        'tf_gpu': ['tensorflow-gpu>=1.6.0'],
    },
    keywords=['OMR', 'Page content detection', 'pixel classifier'],
    data_files=[('', ["requirements.txt"])],
)
