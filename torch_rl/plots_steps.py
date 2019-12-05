import bottleneck as bn
bn.__version__ = '1.2.1'
import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_style("ticks")

params = {'legend.fontsize': 10, 'legend.handlelength': 2,
          'font.size': 10}
plt.rcParams.update(params)

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    print(values)
    print(sma)
    return sma

agg_fn = np.mean
# plotname = "Paper_N3r.pdf"
window_size = 10
plot_std_err = True
# ylims = (5.5, 10)
# ylims = (4, 10)
xlims = None
max_step = 150000

env_type = "stochastic_4_channels" # "stochastic" # "deterministic"

path = 'storage/{}/'
plotname = 'storage/{}/plot.pdf'.format(env_type)

experiments = {
     "PPO": ["{}/ppo_{}".format(env_type, i) for i in range(2, 4)],
     "PPO+VAE": ["{}/ppo_vae_{}".format(env_type, i) for i in range(2, 4)],
     "PPO+Transition": ["{}/ppo_transition_{}".format(env_type, i) for i in range(2, 4)],
     "PPO+VAE+Transition": ["{}/ppo_vae_transition_{}".format(env_type, i) for i in range(2, 4)]
}


fig_main, ax_main = plt.subplots(1,1)
palette = sns.color_palette()

for key_idx, key in enumerate(experiments):
    print(key)
    all_steps = []
    all_values = []

    all_eval_steps = []
    all_eval_values = []

    for idx in range(len(experiments[key])):
        print(idx)
        dirname = experiments[key][idx]
        print(dirname)
        steps = []
        values = []
        eval_steps = []
        eval_values = []
        modified_path = path.format(dirname)
        for filename in os.listdir(modified_path):
            if filename.startswith('log'):

                try:
                    # Read evaluation results.
                    f = open(modified_path + 'log.txt')
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if 'Eval Episodes:' in line:
                            eval_line = lines[i+1]
                            frames = round(int(lines[i-2].split('|')[1].split(' ')[2]), -4)
                            eval_steps.append(frames)
                            average_steps = float(eval_line.split('|')[-1].split(' ')[2])
                            eval_values.append(average_steps)
                except:
                    pass

            if not filename.startswith('events'):
                continue
            try:
                print(modified_path + filename)
                for e in tf.compat.v1.train.summary_iterator(modified_path + filename):
                    for v in e.summary.value:
                        if v.tag == 'num_frames_mean' and e.step <= max_step:
                            steps.append(e.step)
                            values.append(v.simple_value)
            except:
                pass
            # print(e)

        steps = np.array(steps)[window_size//2:-window_size//2]
        values = movingaverage(np.array(values), window_size)
        min_len = min(steps.shape[0], values.shape[0])
        values, steps = values[:min_len], steps[:min_len]

        all_steps.append(steps)
        all_values.append(values)

        eval_steps = np.array(eval_steps)
        eval_values = np.array(eval_values)
        min_len = min(eval_steps.shape[0], eval_values.shape[0])
        eval_values, eval_steps = eval_values[:min_len], eval_steps[:min_len]

        all_eval_steps.append(eval_steps)
        all_eval_values.append(eval_values)
        print(all_eval_values)
        print('')
        print('')
        print('')

    min_length = np.inf
    for steps, values in zip(all_steps, all_values):
        min_length = min(min_length, steps.shape[0])
        min_length = min(min_length, values.shape[0])
    new_all_steps = []
    new_all_values = []
    for steps, values in zip(all_steps, all_values):
        new_all_steps.append(steps[:min_length])
        new_all_values.append(values[:min_length])
    all_steps = np.stack(new_all_steps)
    all_values = np.stack(new_all_values)

    min_length = np.inf
    for steps, values in zip(all_eval_steps, all_eval_values):
        min_length = min(min_length, steps.shape[0])
        min_length = min(min_length, values.shape[0])
    new_all_eval_steps = []
    new_all_eval_values = []
    for steps, values in zip(all_eval_steps, all_eval_values):
        new_all_eval_steps.append(steps[:min_length])
        new_all_eval_values.append(values[:min_length])
    all_eval_steps = np.stack(new_all_eval_steps)
    all_eval_values = np.stack(new_all_eval_values)

    num_trajectories = all_values.shape[0]
    print("Number trajectoriesf: {}".format(num_trajectories))
    mean = agg_fn(all_values, 0)[::1]
    std = np.std(all_values, 0)[::1] / np.sqrt(num_trajectories)
    steps = all_steps[0][::1]
    print(mean.shape)
    ax_main.plot(steps, mean, label=key, color=palette[key_idx])
    if plot_std_err:
        # ax_main.fill_between(steps, mean+std, mean-std, alpha=0.5, color=palette[key_idx])
        ax_main.fill_between(steps, mean+std, mean-std, alpha=0.5, color=palette[key_idx])

    print(all_values)
    print(all_eval_values)
    eval_mean = agg_fn(all_eval_values, 0)[::1]
    eval_std = agg_fn(all_eval_values, 0)[::1] / np.sqrt(all_eval_values.shape[0])
    # ax_main.plot(all_eval_steps[0][::1], eval_mean, '--', label=key + ' eval', color=palette[key_idx])


ax_main.legend(loc='upper right')
ax_main.set_xlabel("Frames")
ax_main.set_ylabel("Episode Steps")
# ax_main.set_ylim(*ylims)
# if xlims is not None:
#     ax_main.set_xlim(*xlims)
fig_main.savefig(plotname, bbox_inches='tight')


