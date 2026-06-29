import os
import os.path as osp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tyro
from plot_args import Args
from datetime import datetime

args = tyro.cli(Args)

def get_data(logdir, ts, grid_res):
    datasets = []
    uniform_grid = np.linspace(0, ts, num=grid_res)
    for root, _, files in os.walk(logdir):
        if 'experimental-setup' in files:
            exp_name = None
            config_path = osp.join(root, "experimental-setup")
            try:
                with open(config_path, "rb") as f:
                    config = eval(f.read())
                if config['Notes'] == 'UDRM':
                    exp_name = config['Notes'] + "_" + str(config['alpha_int'])
                else:
                    exp_name = config['Notes']
            except Exception as e:
                print(f"Error reading {config_path}: {type(e).__name__}: {e}")
                continue  # Skip this directory if config fails

            try:
                file_path = open(osp.join(root, "log_data.pkl"), "rb")
                log_data = pickle.load(file_path)
            except Exception as e:
                print(f"Could not read from {osp.join(root, 'log_data.pkl')}: {e}")
                continue  # Skip this directory if log_data fails
            
            score_data = np.array(log_data['Episode_score'])
            orig_scores = score_data[:,0]
            timesteps = score_data[:,1]
            interpolated_y = np.interp(uniform_grid, timesteps, orig_scores)
            exp_data = pd.DataFrame({'Timestep': uniform_grid, 'Score': interpolated_y})
            
            exp_data.insert(len(exp_data.columns), 'Notes', exp_name)
            datasets.append(exp_data)
    return datasets

def get_all_data(all_logdirs):
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir.endswith("/"):
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    data = []
    for log in logdirs:
        data += get_data(log, args.n_timesteps, args.grid_res)
    return data

def plot(data, xaxis='Timestep', value='Score', game='Asterix',
         c='Notes', smooth=1, estimator='mean'):
    if smooth > 1:
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x
    # plt.figure()
    fig, ax = plt.subplots()
    ax.set_facecolor("#EAEAF2") # This is the exact hex code Seaborn uses for darkgrid
    ax.grid(True, color='white')
    ax.set_axisbelow(True)
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set_theme(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=c, 
                errorbar=('se', 1.96), estimator=estimator, linewidth=0.8, ax=ax)
    plt.legend(loc='upper center', ncol=5, handlelength=1, 
            mode="expand", borderaxespad=0., prop={'size': 7})
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')      
        spine.set_linewidth(0.5)
    plt.title(game)
    plt.xlabel("Timestep", fontsize=10)
    plt.ylabel("Reward", fontsize=10)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.gca().xaxis.offsetText.set_fontsize(10)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout(pad=0.5)
    # plt.show()
    return plt

filedir = os.path.realpath(__file__)
folder_idx = filedir.rfind("/")
folder_dir = filedir[:folder_idx+1]
root_folder = folder_dir + "plots/" + datetime.now().strftime('%Y-%m-%d-%H%M%S') + "_results"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

# print(args.logdir)
data = get_all_data(args.logdir)
# print(f"data: {data}")
fig = plot(data, xaxis="Timestep", value="Score", game=args.game, c="Notes", smooth=11, estimator='mean')

plot_file = os.path.join(root_folder, f"{args.game}.png")
fig.savefig(plot_file, dpi=500)
print(f"Outputs saved to: {root_folder}")
