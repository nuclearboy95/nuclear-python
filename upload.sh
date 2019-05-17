#!/bin/bash

rm dist/*
python setup.py bdist_wheel
twine upload dist/nuclear_python-*.whl