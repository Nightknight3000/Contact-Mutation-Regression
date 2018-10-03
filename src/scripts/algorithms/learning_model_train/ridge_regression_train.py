import logging

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso  # benötigt alpha=0.0001 Parameter
from sklearn.linear_model import ElasticNet  # benötigt alpha=0.0001 Parameter
from sklearn.linear_model import LassoLars  # benötigt alpha=0.0001 Parameter
from sklearn.linear_model import HuberRegressor  # benötigt alpha=0.0001 Parameter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor  # benötigt alpha=0.0001 Parameter [verworfen]
from sklearn.tree import DecisionTreeRegressor  # [verworfen]
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression  # [verworfen]
from sklearn.neural_network import MLPRegressor  # benötigt alpha=0.0001 Parameter

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("Ridge regression training")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


def train_ridge_regression(dataset_train, ddG_train):
    LOG.info("Start ridge regression training")
    regression = MLPRegressor()
    regression = regression.fit(dataset_train, ddG_train)
    LOG.info("Finished ridge regression training")
    return regression
