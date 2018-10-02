import numpy as np
import matplotlib.pyplot as plt

from src.scripts.io.writer.plot_file_writer import write_plot_as_png


def setup_evaluation_plot(prediction, ddG_test, classifier_tool, inputfile):
    title = inputfile.split('/')[len(inputfile.split('/')) - 1].replace('.csv', '') + '_' + classifier_tool
    differences = [abs(ddG_test.values.tolist()[i] - prediction[i]) for i in range(len(prediction))]

    # figure 1: difference_figure
    plot_difference_figure(differences, title)

    # figure 2: pred_and_test_plot_figure
    plot_pred_and_test_plot_figure(prediction, ddG_test, title)

    # figure 3: pred_and_test_point_figure
    plot_pred_and_test_point_figure(prediction, ddG_test, title)


def plot_difference_figure(differences, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(differences, '-b', label='difference (mean=' + str(round(np.array(differences).mean(), 2)) + ')')
    plt.legend(loc='upper right')
    plt.ylabel('ddG')
    write_plot_as_png(plt, title + '01.png')


def plot_pred_and_test_plot_figure(prediction, ddG_test, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(prediction, '-b', label='pred_value')
    plt.plot(ddG_test.values.tolist(), '-r', label='test_value')
    plt.legend(loc='upper right')
    plt.ylabel('ddG')
    write_plot_as_png(plt, title + '02.png')


def plot_pred_and_test_point_figure(prediction, ddG_test, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(prediction, ddG_test.values.tolist(), '.', label='')
    plt.plot([-5, 0, 5], [-5, 0, 5], '--', label='id')
    plt.legend(loc='upper right')
    plt.xlabel('pred_val')
    plt.ylabel('test_val')
    write_plot_as_png(plt, title + '03.png')
