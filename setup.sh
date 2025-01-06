#!/bin/bash

rm dist/*
python setup.py sdist
pip install dist/nuclear_python-*.tar.gz
