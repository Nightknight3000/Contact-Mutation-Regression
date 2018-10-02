import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
LOG = logging.getLogger("K-Fold-Crossvalidation")
LOG.addHandler(console)
LOG.setLevel(logging.INFO)


def k_fold_cross_validation(regression, dataset_test, ddG_test, inputfile):
    splits = 10
    LOG.info("Perform "+str(splits)+"-fold-crossvalidation")
    prediction = regression.predict(dataset_test)
    differences = [abs(ddG_test.values.tolist()[i]-prediction[i]) for i in range(len(prediction))]
    write_plots(differences, prediction, ddG_test, inputfile)
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


def write_plots(differences, prediction, ddG_test, inputfile):
    # figure 1: difference_figure
    inputfilename = inputfile.split('/')[len(inputfile.split('/'))-1].replace('.csv', '')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(differences, '-b', label='difference (mean=' + str(round(np.array(differences).mean(), 2)) + ')')
    plt.legend(loc='upper right')
    plt.ylabel('ddG')
    plt.savefig('data/plots/difference_figure/'+inputfilename+'_01.png')

    # figure 2: pred_and_test_plot_figure
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(prediction, '-b', label='pred_value')
    plt.plot(ddG_test.values.tolist(), '-r', label='test_value')
    plt.legend(loc='upper right')
    plt.ylabel('ddG')
    plt.savefig('data/plots/pred_and_test_plot_figure/'+inputfilename+'_02.png')

    # figure 3: pred_and_test_point_figure
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(prediction, ddG_test.values.tolist(), '.', label='')
    plt.plot([-5, 0, 5], [-5, 0, 5], '--', label='id')
    plt.legend(loc='upper right')
    plt.xlabel('pred_val')
    plt.ylabel('test_val')
    plt.savefig('data/plots/pred_and_test_point_figure/'+inputfilename+'_03.png')
