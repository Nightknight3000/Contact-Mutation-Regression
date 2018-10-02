import click
import logging
import sys

from src.scripts.io.parser.csv_blomap_parser import parse_csv_blomap
from src.scripts.algorithms.dataset.random_split import random_dataset_split
from src.scripts.algorithms.learning_model_train.gbtree_regression_train import train_xgb_regression
from src.scripts.algorithms.learning_model_train.sklearn_regression_train import train_sklearn_regression
from src.scripts.algorithms.learning_model_train.ridge_regression_train import train_ridge_regression
from src.scripts.algorithms.evaluation.cross_validation import k_fold_cross_validation
from src.scripts.algorithms.evaluation.setup_eval_plot import setup_evaluation_plot
from src.scripts.algorithms.learning_model_predict.learning_model_predict import predict_using_regression
from src.scripts.io.writer.predicted_file_writer import write_predicted_blomap_file

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("Commandline parser")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


@click.command()
@click.option('-c', '--classifier_tool', prompt='choose classifier tool',
              help='choose classifier tool (either: xgb, sklearn, ridge', required=True)
@click.option('-i', '--input_training_map', prompt='input training contact map',
              help='input path to contact map to train regression', required=True)
@click.option('-p', '--input_prediction_map', prompt='input prediction contact map',
              help='input path to contact map to be predicted by regression')
@click.option('-o', '--output_predicted_map', prompt='output predicted contact map',
              help='output path for predicted contact map')
@click.option('-s/-v', '--silent/--verbose', default=True)
def main(classifier_tool, input_training_map, input_prediction_map, output_predicted_map, silent):
    LOG.info("Start Contact-Mutation-Regression")
    allowed_classifiers = ['xgb', 'sklearn', 'ridge']
    if input_training_map.lower().endswith(('.csv', '.txt')) and classifier_tool in allowed_classifiers:

        input_dataset = parse_csv_blomap(input_training_map, True)
        dataset_train, dataset_test, ddG_train, ddG_test = random_dataset_split(input_dataset)

        if classifier_tool == 'xgb':
            regression = train_xgb_regression(dataset_train, ddG_train, silent)
        elif classifier_tool == 'sklearn':
            regression = train_sklearn_regression(dataset_train, ddG_train)
        else:
            regression = train_ridge_regression(dataset_train, ddG_train)

        k_fold_cross_validation(regression, dataset_test, ddG_test)
        prediction = regression.predict(dataset_test)
        setup_evaluation_plot(prediction, ddG_test, classifier_tool, input_training_map)

        if input_prediction_map != 'None':
            if input_prediction_map.lower().endswith(('.csv', '.txt')) and output_predicted_map != 'None':
                predicted_contact_map = predict_using_regression(regression,
                                                                 parse_csv_blomap(input_prediction_map, False))
                write_predicted_blomap_file(predicted_contact_map, output_predicted_map)
            else:
                LOG.error('Please check your inputfile extension to be either .txt. or .csv, ' +
                          'and specify an outputfilepath!')
    else:
        LOG.error('Inputfile invalid extension error')

    LOG.info("End Contact-Mutation-Regression")
    sys.exit()
