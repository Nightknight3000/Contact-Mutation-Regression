import logging

from sklearn.linear_model import Ridge

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("Ridge regression training")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


def train_ridge_regression(dataset_train, ddG_train):
    LOG.info("Start ridge regression training")
    regression = Ridge()
    regression = regression.fit(dataset_train, ddG_train)
    LOG.info("Finished ridge regression training")
    return regression
