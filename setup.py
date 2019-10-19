from setuptools import setup, find_packages
from glob import glob

with open('requirements.txt', 'r') as f:
    install_reqs = [
        s for s in [
            line.strip(' \n') for line in f
        ] if not s.startswith('#') and s != ''
    ]

setup(name='nuclear-python',

      version='0.11.8',

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
      install_requires=install_reqs,
      test_suite='nose.collector')
