#!/bin/sh
export jexec=/home/adam/jython2.7.0/bin/jython 
$jexec NN-Backprop.py
$jexec NN-GA.py
$jexec NN-RHC.py
$jexec NN-SA.py
$jexec continuouspeaks.py
$jexec flipflop.py
$jexec tsp.py
