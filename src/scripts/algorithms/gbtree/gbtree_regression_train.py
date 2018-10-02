import logging

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("Regression training")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


def train_regression(dataset_train, ddG_train, silent):
    LOG.info("Start regression training")
    # regression = XGBRegressor(silent=silent)
    regression = LinearRegression()
    regression = regression.fit(dataset_train, ddG_train)
    LOG.info("Finished regression training")
    return regression
