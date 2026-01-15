import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Процесс обучения')
    plt.xlabel('Количество игр')
    plt.ylabel('Счет')
    plt.plot(scores, label='Текущий счет')
    plt.plot(mean_scores, label='Средний счет')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)
