from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=1.8.0', 'scipy>=1.1.0', 'sh>=1.12.14', 'numpy>=1.14.3', 'sklearn>=0.19.1', 'pandas>=0.22.0']

setup(
	name = 'trainer',
	version = '0.2',
	install_requires = REQUIRED_PACKAGES,
	packages = find_packages(),
	include_package_data = True,
	description = 'Recommender System'
)
