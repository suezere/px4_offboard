import os
from glob import glob
from setuptools import setup

package_name = 'px4_offboard'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/ament_index/resource_index/packages',
            ['resource/' + 'visualize.rviz']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('resource/*rviz'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Howard Li',
    maintainer_email='lch200051@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'offboard = px4_offboard.offboard:main',
                'offboard_multi = px4_offboard.offboard_multi:main',
                'visualizer = px4_offboard.visualizer:main',
                'test_offboard_multi = px4_offboard.test_offboard_multi:main',
                'test_offboard = px4_offboard.test_offboard:main',
                'offboard_control = px4_offboard.offboard_control:main',
                'processes = px4_offboard.processes:main'
        ],
    },
)
