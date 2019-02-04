import argparse
from datetime import datetime
import logging
import numpy as np

import experiments
from data import loader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_details, experiment, timing_key, verbose, timings):
    t = datetime.now()
    for details in experiment_details:
        exp = experiment(details, verbose=verbose)

        logger.info("Running {} experiment: {}".format(timing_key, details.ds_readable_name))
        exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform some SL experiments')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--ann', action='store_true', help='Run the ANN experiment')
    parser.add_argument('--boosting', action='store_true', help='Run the Boosting experiment')
    parser.add_argument('--dt', action='store_true', help='Run the Decision Tree experiment')
    parser.add_argument('--knn', action='store_true', help='Run the KNN experiment')
    parser.add_argument('--svm', action='store_true', help='Run the SVM experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    seed = 42
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        print("Using seed {}".format(seed))

    print("Loading data")
    print("----------")

    ds1_details = {
            'data': loader.WineData(verbose=verbose, seed=seed),
            'name': 'wine',
            'readable_name': 'Wine Data (Red/White)',
        }
    ds2_details = {
            'data': loader.GenderVoiceData(verbose=verbose, seed=seed),
            'name': 'gvd',
            'readable_name': 'Gender Voice Data',
        }
    wine_quality_details = {
            'data': loader.WineQuality(verbose=verbose, seed=seed),
            'name': 'wine-qual',
            'readable_name': 'Wine Data (Quality)',
        }
    wine_quality_uniq_details = {
            'data': loader.WineQualityUniq(verbose=verbose, seed=seed),
            'name': 'wine-qual',
            'readable_name': 'Wine Data (Quality)',
        }

    enhancer_human_heart = {
            'data': loader.EnhancerHumanHeart(verbose=verbose, seed=seed),
            'name': 'enhancer-hh',
            'readable_name': 'Genomic Enhancer (Human Heart)',
        }
    enhancer_brain = {
            'data': loader.EnhancerBrain(verbose=verbose, seed=seed),
            'name': 'enhancer-b',
            'readable_name': 'Genomic Enhancer (Brain)',
        }


    if verbose:
        print("----------")
    print("Running experiments")

    timings = {}

    datasets = [
        ds1_details,
        ds2_details
    ]
    datasets = [enhancer_brain, wine_quality_details]
    datasets = [wine_quality_uniq_details]
    datasets = [enhancer_brain, wine_quality_uniq_details]
    experiment_details = []
    for ds in datasets:
        data = ds['data']
        data.load_and_process()
        data.build_train_test_split()
        data.scale_standard()
        experiment_details.append(experiments.ExperimentDetails(
            data, ds['name'], ds['readable_name'],
            threads=threads,
            seed=seed, bparams=False, # Turn this to True for best params in each clf
        ))

    if args.knn or args.all:
        run_experiment(experiment_details, experiments.KNNExperiment, 'KNN', verbose, timings)

    if args.boosting or args.all:
        run_experiment(experiment_details, experiments.BoostingExperiment, 'Boosting', verbose, timings)

    if args.ann or args.all:
        run_experiment(experiment_details, experiments.ANNExperiment, 'ANN', verbose, timings)

    if args.dt or args.all:
        run_experiment(experiment_details, experiments.DTExperiment, 'DT', verbose, timings)


    if args.svm or args.all:
        run_experiment(experiment_details, experiments.SVMExperiment, 'SVM', verbose, timings)

    print(timings)
