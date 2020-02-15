import matplotlib.pyplot as plt
import torch


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


def plot_validation_score(validation_score_list, episodes_list, fig_num=3, y_label='Q value'):
    plt.figure(fig_num)
    plt.clf()
    validation_score_list_t = torch.tensor(validation_score_list, dtype=torch.float)
    plt.title('Validation')
    plt.xlabel('Episode')
    plt.ylabel(y_label)
    plt.plot(episodes_list ,validation_score_list_t.numpy())
    # Take 100 episode averages and plot them too
    plt.pause(0.001)  # pause a bit so that plots are updated
