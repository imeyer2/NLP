#!/bin/bash

total_sgd=`ls -lh COMBINED3/TEST_sgd* | wc -l`
total_bfgs=`ls -lh COMBINED3/TEST_bfgs* | wc -l`
total_fr=`ls -lh COMBINED3/TEST_fr* | wc -l`

echo the total sgd experiments in the directory is $total_sgd
echo the total fr experiments in the directory is $total_fr
echo the total bfgs experiments in the directory is $total_bfgs
