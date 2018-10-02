import logging
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("K-Fold-Crossvalidation")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


def k_fold_cross_validation(regression, dataset_test, ddG_test):
    splits = 10
    LOG.info("Perform "+str(splits)+"-fold-crossvalidation")
    k_fold = cross_val_score(regression, np.asarray(dataset_test), np.asarray(ddG_test), cv=splits,
                             scoring="neg_mean_absolute_error")
    # print("hi "+str(len(prediction.tolist()))+'\n'+str(prediction.tolist())+"\n")
    # print("HALLO "+str(len(ddG_test.values.tolist()))+'\n'+str(ddG_test.values.tolist()))
    # LOG.info("Finished "+str(splits)+"-fold-crossvalidation")
    # LOG.info('Mean-Absolute-Error: '+str(round(mean_absolute_error(ddG_test.values.tolist(), prediction), 2)))
    # LOG.info('Mean-Squared-Error: '+str(round(mean_squared_error(ddG_test.values.tolist(), prediction), 2)))
    # LOG.info('CV-Accuracy-Mean: '+str(round(k_fold.mean()*100, 2))+'%')
    # LOG.info('CV-Standard-Deviation: '+str(round(k_fold.std(), 2)))
    # LOG.info('ddG-Interval: ['+str(ddG_test.min())+', '+str(ddG_test.max())+'] => Intervalsize: '+str(round(ddG_test.max()-ddG_test.min(), 2)))
    # LOG.info('ddG-Deviation-Percentage: '+str(round(k_fold.std()*100/ddG_test.max()-ddG_test.min(), 2))+"%")

