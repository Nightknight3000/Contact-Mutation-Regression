import click
import logging
import sys

from src.scripts.io.parser.contact_map_parser import csv_contact_map_parser
from src.scripts.algorithms.dataset.random_split import random_dataset_split
from src.scripts.algorithms.gbtree.gbtree_regression_train import train_regression
from src.scripts.algorithms.evaluation.cross_validation import k_fold_cross_validation
from src.scripts.algorithms.gbtree.gbtree_regression_predict import predict_using_regression
from src.scripts.io.writer.write_predictionfile import write_predicted_contact_map_file

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("Commandline parser")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


@click.command()
@click.option('-i', '--input_training_map', prompt='input training contact map',
              help='input path to contact map to train regression', required=True)
@click.option('-p', '--input_prediction_map', prompt='input prediction contact map',
              help='input path to contact map to be predicted by regression')
@click.option('-o', '--output_predicted_map', prompt='output predicted contact map',
              help='output path for predicted contact map')
@click.option('-s/-v', '--silent/--verbose', default=True)
def main(input_training_map, input_prediction_map, output_predicted_map, silent):
    LOG.info("Start Contact-Mutation-Regression")
    if input_training_map.lower().endswith(('.csv', '.txt')):

        input_dataset = csv_contact_map_parser(input_training_map, True)
        dataset_train, dataset_test, ddG_train, ddG_test = random_dataset_split(input_dataset)

        regression = train_regression(dataset_train, ddG_train, silent)
        k_fold_cross_validation(regression, dataset_test, ddG_test, input_training_map)

        if input_prediction_map is True and input_training_map.lower().endswith(('.csv', '.txt')) and output_predicted_map is True:
            predicted_contact_map = predict_using_regression(regression,
                                                             csv_contact_map_parser(input_prediction_map, False))
            write_predicted_contact_map_file(predicted_contact_map, output_predicted_map)
    else:
        LOG.error('Input error')

    LOG.info("End Contact-Mutation-Regression")
    sys.exit()
