from matplotlib import pyplot as plt
import math

def plot_nnz_vs_accuracy(data_policy, data_pruned, legend=('policy_dist', 'PDX2'),
                         title='NNZ_vs_Accuracy', xlabel='NNZ0', ylabel='accuracy'):
    fig = plt.figure()
    x_policy = data_policy[0][:]
    x_pruned = data_pruned[0][:]
    acc_policy = data_policy[1][:]
    acc_pruned = data_pruned[1][:]
    plt.plot(x_policy, acc_policy, marker='o', color='b')
    plt.plot(x_pruned, acc_pruned, marker='^', color='g')
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()
    fig.savefig('results.png')

def plot_nnz_vs_accuracy_latest(data_policy, data_pruned, data_PDX2):
    fig = plt.figure()
    x = data_policy[0][:]
    acc_policy = data_policy[1][:]
    acc_pruned = data_pruned[1][:]
    acc_PDX2 = data_PDX2[1][:]
    plt.plot(x, acc_policy, marker='o', color='b')
    plt.plot(x, acc_pruned, marker='^', color='g')
    plt.plot(x, acc_PDX2, marker='*', color='r')
    plt.legend(('policy_dist', 'pruning', 'PDX2'))
    plt.xlabel('NNZ')
    plt.ylabel('accuracy')
    plt.title('NNZ_vs_Accuracy')
    plt.grid()
    plt.show()
    fig.savefig('results.png')

def plot_weights(agent, title: str, figure_num: int , range=5):
    weights_matrices = agent.sess.run(agent.weights_matrices)
    plot_histogram(weights_matrices, title, include_zeros=False, figure_num=figure_num, range=(-range, range))


def plot_histogram(weights_list: list,
                   image_name: str,
                   range: tuple,
                   include_zeros=True,
                   figure_num=1):

    """A function to plot weights distribution"""

    weights = []
    for w in weights_list:
        weights.extend(list(w.ravel()))

    if not include_zeros:
        weights = [w for w in weights if w != 0]

    fig = plt.figure(num=figure_num, figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.hist(weights,
            bins=100,
            facecolor='green',
            edgecolor='black',
            alpha=0.7,
            range=range)

    ax.set_title('Weights distribution \n ' + image_name)
    ax.set_xlabel('Weights values')
    ax.set_ylabel('Number of weights')

    fig.savefig(image_name + '.png')


def plot_graph(data, name: str, figure_num=1, file_name=None, xaxis='sparsity', yaxis='accuracy'):
  fig = plt.figure(figure_num)
  x = data[0]
  y = data[1]
  plt.plot(x[:], y[:], 'ro')
  plt.xlabel(xaxis)
  plt.ylabel(yaxis)
  plt.title(name)
  plt.grid()
  filename = name if file_name is None else file_name
  fig.savefig(filename + '.png')

def plot_conv_weights(model, title='weights', figure_num=1):
    weights = model.get_flat_weights()
    plot_histogram(weights_list=weights, image_name=title, include_zeros=False, range=(-1.0, 1.0), figure_num=figure_num)
