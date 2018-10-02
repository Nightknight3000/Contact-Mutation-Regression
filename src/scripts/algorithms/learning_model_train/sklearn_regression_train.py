import logging

from sklearn.linear_model import LinearRegression

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("Sklearn regression training")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


def train_sklearn_regression(dataset_train, ddG_train):
    LOG.info("Start sklearn regression training")
    regression = LinearRegression
    regression = regression.fit(dataset_train, ddG_train)
    LOG.info("Finished sklearn regression training")
    return regression
