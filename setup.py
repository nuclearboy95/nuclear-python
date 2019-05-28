from setuptools import setup, find_packages
from glob import glob

setup(name='nuclear-python',

      version='0.9.2',

      url='https://github.com/nuclearboy95/nuclear-python',

      license='MIT',

      author='Jihun Yi',
      scripts=glob('bash_scripts/*'),
      author_email='t080205@gmail.com',

      description='Python helpers',

      packages=find_packages(exclude=['tests', 'dist', 'build']),
      include_package_data=True,

      long_description=open('README.md').read(),

      zip_safe=False,

      setup_requires=['nose>=1.0'],

      test_suite='nose.collector')
