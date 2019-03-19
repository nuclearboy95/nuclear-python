#!/bin/bash

rm dist/*
python setup.py sdist
pip install dist/nuclear-python-*.tar.gz
