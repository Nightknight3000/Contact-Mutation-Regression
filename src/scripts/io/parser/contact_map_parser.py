import logging
import pandas as pd

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("Contact map parser")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


def csv_contact_map_parser(input_csv_file, is_training_set):
    LOG.info("Parse contact map, setup datasets")
    training_dataset = pd.read_csv(input_csv_file, delimiter=',', header=None)
    input_data = [training_dataset.iloc[:, :training_dataset.shape[1]-1]]
    if is_training_set:
        input_data.append(training_dataset.iloc[:, training_dataset.shape[1]-1])
    LOG.info("Finished parsing")
    return input_data
