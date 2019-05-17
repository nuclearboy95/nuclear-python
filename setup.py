from setuptools import setup, find_packages

setup(name='nuclear-python',

      version='0.6',

      url='https://github.com/nuclearboy95/nuclear-python',

      license='MIT',

      author='Jihun Yi',

      author_email='t080205@gmail.com',

      description='Python helpers',

      packages=find_packages(exclude=['tests']),
      include_package_data=True,

      long_description=open('README.md').read(),

      zip_safe=False,

      setup_requires=['nose>=1.0'],

      test_suite='nose.collector')
