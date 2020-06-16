#!/bin/bash

DATASET="cora"
NUMP=3

for node in $(cat ../targets/$DATASET.txt)
do 
   python defense.py --target $node --dataset $DATASET  -p $NUMP -j 0.02 -r 10
done
