import os
from glob import glob
from setuptools import setup

package_name = 'aera_semi_autonomous'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages',
                 ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml']),
                (os.path.join('share', package_name, 'launch'),
                 glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
                (os.path.join('share', package_name, 'config'),
                 glob(os.path.join('config', '*.yaml')))],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Wiktor Jurasz',
    author_email='wiktor.jurasz@gmail.com',
    maintainer='Wiktor Jurasz',
    maintainer_email='wiktor.jurasz@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Package to run AR4 arm using one-shot semi autonomous pipeline',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aera_semi_autonomous_node = aera_semi_autonomous.aera_semi_autonomous_node:main',
        ],
    },
)
