#!/bin/sh

# If you want to first generate data and updated datasets, remove the "--skiprerun" flags below

export ica_wine=4
export ica_enhancer=7
export pca_wine=3
export pca_enhancer=4
export rp_wine=6
export rp_enhancer=7
export rf_wine=3
export rf_enhancer=5



#python run_experiment.py --all --verbose --threads -1  > run-all.log 2>&1
python run_experiment.py --ica --dataset1 --dim $ica_wine     --skiprerun --verbose --threads -1 > ica-wine-clustering.log 2>&1
python run_experiment.py --ica --dataset2 --dim $ica_enhancer --skiprerun --verbose --threads -1 > ica-enhancer-clustering.log   2>&1
python run_experiment.py --pca --dataset1 --dim $pca_wine     --skiprerun --verbose --threads -1 > pca-wine-clustering.log 2>&1
python run_experiment.py --pca --dataset2 --dim $pca_enhancer --skiprerun --verbose --threads -1 > pca-enhancer-clustering.log   2>&1
python run_experiment.py --rp  --dataset1 --dim $rp_wine      --skiprerun --verbose --threads -1 > rp-wine-clustering.log  2>&1
python run_experiment.py --rp  --dataset2 --dim $rp_enhancer  --skiprerun --verbose --threads -1 > rp-enhancer-clustering.log    2>&1
python run_experiment.py --rf  --dataset1 --dim $rf_wine      --skiprerun --verbose --threads -1 > rf-wine-clustering.log  2>&1
python run_experiment.py --rf  --dataset2 --dim $rf_enhancer  --skiprerun --verbose --threads -1 > rf-enhancer-clustering.log    2>&1
#python run_experiment.py --svd --dataset1 --dim X  --skiprerun --verbose --threads -1 > svd-wine-clustering.log 2>&1
#python run_experiment.py --svd --dataset2   --dim X  --skiprerun --verbose --threads -1 > svd-enhancer-clustering.log   2>&1
python run_experiment.py --plot

