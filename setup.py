from setuptools import setup, find_packages

setup(name='rbeis',
      version='0.0.1',
      description='RBEIS implementation for Pandas and PySpark',
      author='Phil Yeeles',
      author_email='Phil.Yeeles@ons.gov.uk',
      packages=find_packages(),
      install_requires=['numpy','pandas'],
      test_suite='tests',
)
