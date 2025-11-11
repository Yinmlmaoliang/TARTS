from setuptools import setup
import os
from glob import glob

package_name = 'tarts_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, 'tarts_ros.footprint_utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TARTS Maintainer',
    maintainer_email='user@example.com',
    description='ROS2 wrapper for TARTS (Template-Assisted Reference-based Target Segmentation)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tarts_segmentation = tarts_ros.segmentation_node:main',
            'tarts_register_prototype = tarts_ros.register_prototype_node:main',
            'tarts_prototype_update = tarts_ros.prototype_update_node:main',
        ],
    },
)
