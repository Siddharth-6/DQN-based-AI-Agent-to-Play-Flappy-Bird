import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def plot_survival(survival_times, mean_survival_times):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Survival Time Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Steps Survived')
    plt.plot(survival_times, label='Survival Time', alpha=0.7)
    plt.plot(mean_survival_times, label='Mean Survival Time', linewidth=2)
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(survival_times)-1, survival_times[-1], str(survival_times[-1]))
    plt.text(len(mean_survival_times)-1, mean_survival_times[-1], str(int(mean_survival_times[-1])))
    plt.show(block=False)
    plt.pause(.1)