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
                continue

            try:
                file_path = open(osp.join(root, "log_data.pkl"), "rb")
                log_data = pickle.load(file_path)
            except Exception as e:
                print(f"Could not read from {osp.join(root, 'log_data.pkl')}: {e}")
                continue
            
            score_data = np.array(log_data['Episode_score'])
            orig_scores = score_data[:,0]
            timesteps = score_data[:,1]
            interpolated_y = np.interp(uniform_grid, timesteps, orig_scores)
            exp_data = pd.DataFrame({'Timestep': uniform_grid, 'Score': interpolated_y, 'Seed': config['seed']})
            
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

def plot(games, logdirs, root_folder, xaxis='Timestep', value='Score', c='Notes', smooth=1, estimator='mean'):
    ngames = len(games)
    plot_file = os.path.join(root_folder, f"Other.png")
    if ngames == 1:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(1, ngames, figsize=(20, 5))
    with open(root_folder + '/output_log.txt',"w") as f:
        for idx, game in enumerate(games):
            ndirs = []
            for dir in logdirs:
                if game in dir:
                    ndirs.append(dir)
            data = get_all_data(ndirs)
            if smooth > 1:
                y = np.ones(smooth)
                for datum in data:
                    x = np.asarray(datum[value])
                    z = np.ones(len(x))
                    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                    datum[value] = smoothed_x
            if isinstance(data, list):
                data = pd.concat(data, ignore_index=True)
            sns.set_theme(style="darkgrid", font_scale=1.5)
            seaborn_title_size = plt.rcParams['axes.titlesize']
            seeds_per_model = data.groupby('Notes', sort=False)['Seed'].nunique().to_dict()
            summary = data.groupby('Notes', sort=False)['Score'].agg(['mean', 'std']).to_dict(orient='index')
            print("\n")
            for key in summary.keys():
                f.write(f"\n{key} -> Avg. cumulative reward: {np.round(summary[key]['mean'],4)} ± {np.round(summary[key]['std'],4)}\n")
                print(key, game, seeds_per_model[key])
                log_statement = f"{key} {game} {seeds_per_model[key]}\n"
                f.write(log_statement)
            if ngames == 1:
                ax.set_facecolor("#EAEAF2") # This is the exact hex code Seaborn uses for darkgrid
                ax.grid(True, color='white')
                ax.set_axisbelow(True)
                ax.set_title(game, fontsize=seaborn_title_size)
                ax.tick_params(axis='both', which='major')
                handles, labels = ax.get_legend_handles_labels()
                cols_num = np.ceil(len(all_models)/2)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')      
                    spine.set_linewidth(0.5)
                sns.lineplot(data=data, x=xaxis, y=value, hue=c, 
                        errorbar=('se', 1.96), estimator=estimator, linewidth=0.8, ax=ax)
                ax.legend_.remove()
                ax.set_xlabel("")
                ax.set_ylabel("")
            else:
                ax[idx].set_facecolor("#EAEAF2") # This is the exact hex code Seaborn uses for darkgrid
                ax[idx].grid(True, color='white')
                ax[idx].set_axisbelow(True)
                ax[idx].set_title(game, fontsize=seaborn_title_size)
                ax[idx].tick_params(axis='both', which='major')
                handles, labels = ax[0].get_legend_handles_labels()               
                cols_num = 5
                for spine in ax[idx].spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')      
                    spine.set_linewidth(0.5)
                sns.lineplot(data=data, x=xaxis, y=value, hue=c, errorbar=('se', 1.96), estimator=estimator, linewidth=0.8, ax=ax[idx])
                ax[idx].legend_.remove()
                ax[idx].set_xlabel("")
                ax[idx].set_ylabel("")
            f.write(f"\n-------------------------------------------------------------------------------------------------------------\n")
        fig.tight_layout()
        fig.supxlabel("Timestep", fontsize=10, y=0.1)
        fig.supylabel("Reward", fontsize=10, x=0.002)
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=cols_num)
        plt.subplots_adjust(bottom=0.20)
        plt.show()
        fig.savefig(plot_file, dpi=500)

filedir = os.path.realpath(__file__)
folder_idx = filedir.rfind("/")
folder_dir = filedir[:folder_idx+1]
root_folder = folder_dir + "plots/" + datetime.now().strftime('%Y-%m-%d-%H%M%S') + "_other"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

games = ['Acrobot', 'CartPole', 'FrozenLake', 'MountainCar']
fig = plot(games, args.logdirs, root_folder, xaxis="Timestep", value="Score", c="Notes", smooth=11, estimator='mean')

print(f"\nOutputs saved to: {root_folder}")

