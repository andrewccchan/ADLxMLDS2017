#!/bin/bash
curl -O https://www.csie.ntu.edu.tw/~b03902036/andrew/model.tgz
tar zxvf model.tgz
python3.6 generate.py $1
