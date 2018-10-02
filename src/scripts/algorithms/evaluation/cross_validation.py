import logging
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("K-Fold-Crossvalidation")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


def k_fold_cross_validation(regression, dataset_test, ddG_test):
    splits = 10
    LOG.info("Perform "+str(splits)+"-fold-crossvalidation")
    k_fold = cross_val_score(regression, np.asarray(dataset_test), np.asarray(ddG_test), cv=splits)
    k_fold_mean_absolute_error = cross_val_score(regression, np.asarray(dataset_test), np.asarray(ddG_test), cv=splits,
                             scoring="neg_mean_absolute_error")
    k_fold_mean_squared_error = cross_val_score(regression, np.asarray(dataset_test), np.asarray(ddG_test), cv=splits,
                                                 scoring="neg_mean_squared_error")

    LOG.info("Finished "+str(splits)+"-fold-crossvalidation")
    LOG.info('Mean-Absolute-Error: '+str(round(k_fold_mean_absolute_error.mean(), 2)))
    LOG.info('Mean-Squared-Error: '+str(round(k_fold_mean_squared_error.mean(), 2)))
    LOG.info('CV-Accuracy-Mean: '+str(round(k_fold.mean(), 2)*100)+'%')
    LOG.info('CV-Standard-Deviation: '+str(round(k_fold.std(), 2)))
    LOG.info('ddG-Interval: ['+str(ddG_test.min())+', '+str(ddG_test.max())+'] => Intervalsize: '+str(round(ddG_test.max()-ddG_test.min(), 2)))
    LOG.info('ddG-Deviation-Percentage: '+str(round(k_fold.std()*100/ddG_test.max()-ddG_test.min(), 2))+"%")

    # predicted = cross_val_predict(regression, dataset_test, ddG_test, cv=splits)
    # interval = [np.array(ddG_test.values.tolist()).min(), np.array(ddG_test.values.tolist()).max()]
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 1, 1)
    # plt.plot(predicted, ddG_test.values.tolist(), '.', label='')
    # plt.plot(interval, interval, '--', label='id')
    # plt.legend(loc='upper right')
    # plt.xlabel('pred_val')
    # plt.ylabel('test_val')
    # plt.show()

