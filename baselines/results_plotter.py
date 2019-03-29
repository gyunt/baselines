import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

from baselines.common import plot_util

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_REWARD = 'reward'
Y_TIMESTEPS = 'timesteps'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['darkblue', 'green', 'salmon', 'cyan', 'magenta', 'tan', 'lime', 'coral', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'yellow', 'lightblue', 'lavender', 'red',
          'darkgreen', 'gold', 'darkred', 'turquoise', 'blue', ]


def rolling_window(a, window):
    a = np.concatenate([np.zeros(window - 1), a])

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x, yw_func


def ts2xy(ts, xaxis, yaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
    else:
        raise NotImplementedError
    if yaxis == Y_REWARD:
        y = ts.r.values
    elif yaxis == Y_TIMESTEPS:
        y = ts.l.values
    else:
        raise NotImplementedError
    return x, y


def plot_curves(xy_list, xaxis, yaxis, title):
    fig = plt.figure(figsize=(8, 6))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    # labels = ['ppo_gru_mlp (PR)', 'mlp (PR)', 'ppo_lstm (PR)', 'ppo_gru (PR)', 'ppo_lstm_mlp (PR)', 'mlp (original ppo2)',
    #           'lstm (original ppo2, shared)']
    labels = ['ppo_gru_mlp (PR)', 'mlp (PR)', 'ppo_lstm (PR)', 'ppo_gru (PR)', 'ppo_lstm_mlp (PR)']
    colors = ['darkblue', 'green', 'salmon', 'cyan', 'magenta']

    for (i, (x, y)) in enumerate(xy_list):
        color = colors[i]
        # plt.scatter(x, y, s=2, c= color)
        # x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)  # So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y, color=color, label=labels[i])
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    plt.legend(loc='upper left')
    fig.canvas.mpl_connect('resize_event', lambda event: plt.tight_layout())
    plt.grid(True)


def split_by_task(taskpath):
    return taskpath.dirname.split('/')[-1].split('-')[0]


def plot_results(dirs, num_timesteps=10e6, xaxis=X_TIMESTEPS, yaxis=Y_REWARD, title='', split_fn=split_by_task):
    def a(r):
        return ts2xy(r.monitor, xaxis, 'avg_rewards')

    results = plot_util.load_results(dirs)
    # plot_util.plot_results(results, xy_fn=a, split_fn=split_fn,
    #                        average_group=True, resample=int(1e6))

    xy_list = []

    for result in results:
        x = np.cumsum(result.monitor.l.values)
        y = result.monitor.avg_rewards.values
        xy_list.append([x, y])
    plot_curves(xy_list, "time steps", "avg. last 100 episode rewards", "HalfCheetah-v2")


# Example usage in jupyter-notebook
# from baselines.results_plotter import plot_results
# %matplotlib inline
# plot_results("./log")
# Here ./log is a directory containing the monitor.csv files

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs='*', default=['./log'])
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--xaxis', help='Varible on X-axis', default=X_TIMESTEPS)
    parser.add_argument('--yaxis', help='Varible on Y-axis', default=Y_REWARD)
    parser.add_argument('--task_name', help='Title of plot', default='Breakout')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(dir) for dir in args.dirs]
    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.yaxis, args.task_name)
    # plot_curves()
    plt.show()


if __name__ == '__main__':
    main()
