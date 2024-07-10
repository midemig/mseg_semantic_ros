from setuptools import find_packages, setup

package_name = 'mseg_semantic_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/cfg', 
            ['cfg/default_config_360_ms.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='migueldm',
    maintainer_email='midemig@gmail.com',
    description='TODO: Package description',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sem_seg_node = mseg_semantic_ros.sem_seg_node:main',
        ],
    },
)
