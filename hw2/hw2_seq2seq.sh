#!/bin/bash
curl https://gitlab.com/SimpleA/ADLxMLDS_2017/raw/7e9b8b706f3ac26d2295976888454a5f0d0af4da/best_model.data-00000-of-00001 > best_model.data-00000-of-00001
python test_new.py $1 $2 $3
