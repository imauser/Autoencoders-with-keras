#!/bin/bash
set -e
for file in ./results/*
do
  python ./parserforkeras.py $file
done
