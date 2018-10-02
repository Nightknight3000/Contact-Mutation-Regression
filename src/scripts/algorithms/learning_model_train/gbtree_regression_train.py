import logging

from xgboost import XGBRegressor

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("XGB regression training")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


def train_xgb_regression(dataset_train, ddG_train, silent):
    LOG.info("Start xgb regression training")
    regression = XGBRegressor(silent=silent)
    regression = regression.fit(dataset_train, ddG_train)
    LOG.info("Finished xgb regression training")
    return regression
