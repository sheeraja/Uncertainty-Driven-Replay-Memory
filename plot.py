import os
from scipy.interpolate import UnivariateSpline
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import argparse
from datetime import datetime
         
def plots(n_models, timesteps, rolling_window, subsample, ndirs, ngames, games, min_max, log_folder):
    n_timesteps = timesteps
    sample = subsample
    base_folders = ndirs
    add_min_max = True if min_max == 1 else False
    rolling_window_size = rolling_window
    all_models = []
    
    max_x = n_timesteps / sample

    all_games = games
    
    if ngames == 1:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(1, ngames, figsize=(20, 8))

    plot_file = log_folder + "/minmax_" + str(add_min_max) + '.png'
    n_seeds = 0

    label_by = ["Notes"]
    with open(log_folder + '/output_log.txt',"w") as f:
        for idx, game in enumerate(games):
            model_list = []
            print("\n")
            for base_folder in base_folders:
                folders = list(os.listdir(base_folder))
                label_dict = {}
                score_dict = {}
                stats_dict = {}
                for folder in folders:
                    if game.lower() in folder.lower():
                        info = eval(open(base_folder + folder + '/experimental-setup', 'r').read())
                        label = ""
                        for param in label_by:
                            label += str(info[param])
                        if label not in score_dict:
                            score_dict[label] = []
                            stats_dict[label] = []
                        log_data = pickle.load(open(base_folder + folder + '/log_data.pkl', 'rb'))
                        score_data = np.array(log_data['Episode_score'])
                        scores = score_data[:,0]
                        stats_dict[label].append(np.mean(scores))
                        score_dict[label].append(scores)
                        timesteps = score_data[:,1]

                        all_timesteps = np.arange(0, n_timesteps)
                        spline = UnivariateSpline(timesteps, scores, k=1, s=0, ext=3)
                        scores = spline(all_timesteps)
                        scores = pd.Series(scores)
                        windows = scores.rolling(rolling_window_size)
                        scores = windows.mean()
                        scores = np.where(scores < 0, 0, scores)
                        scores = scores[0:n_timesteps-1:sample]

                        if label in label_dict:
                            label_dict[label].append(scores)
                        else:
                            label_dict[label] = [scores]
                
                for key in stats_dict.keys():
                    all_means = np.mean(stats_dict[key])
                    all_std = np.std(stats_dict[key])
                    f.write(f"\n{key} -> Avg. cumulative reward: {np.round(all_means,4)} ± {np.round(all_std,4)}\n")
                    
                for key in score_dict.keys():
                    score_dict[key] = [reward for seed in score_dict[key] for reward in seed]
                    score_dict[key] = pd.Series(score_dict[key])
                    model_list.append(score_dict[key])
                
                if len(label_dict.keys()) != 0:
                    if list(label_dict.keys())[0] not in all_models:
                        all_models.append(list(label_dict.keys())[0])
                
                for key in label_dict.keys():
                    scores = np.array(label_dict[key])
                    n_seeds = scores.shape[0]
                    mean_scores = np.array(scores).mean(axis=0)
                    x_scaled = np.arange(len(mean_scores)) * sample
                    if ngames == 1:
                        line, = ax.plot(x_scaled, mean_scores, label=key)
                    else:
                        line, = ax[idx].plot(x_scaled, mean_scores, label=key)

                    line_color = line.get_color()
                    if n_seeds > 1:
                        std_scores = np.array(scores).std(axis=0)
                        x_scaled = np.arange(len(std_scores)) * sample
                        lower_bound = mean_scores - 1.96 * std_scores/np.sqrt(n_seeds)
                        upper_bound = mean_scores + 1.96 * std_scores/np.sqrt(n_seeds)
                        
                        if ngames == 1:
                            ax.fill_between(x_scaled, # np.arange(0,scores.shape[1]), 
                                        lower_bound, 
                                        upper_bound, 
                                        alpha=0.2)
                        else:
                            ax[idx].fill_between(x_scaled, # np.arange(0,scores.shape[1]), 
                                        lower_bound, 
                                        upper_bound, 
                                        alpha=0.2)
                    print(key, game, n_seeds)
                    log_statement = f"{key} {game} {n_seeds}\n"
                    f.write(log_statement)
                
                    if add_min_max:
                        data = [x for x in mean_scores if str(x) != 'nan']
                        max_score = max(data)
                        min_score = min(data)
                        x_min_label = len(data) - 1
                        x_max_label = 0
                        if ngames == 1:
                            ax.axhline(y=min_score, color=line_color, linestyle='--')
                            ax.axhline(y=max_score, color=line_color, linestyle='--')
                            ax.text(x_max_label, max_score, f"Max: {max_score:.2f}", color=line_color, va='top', ha='left')
                            ax.text(x_min_label, min_score, f"Min: {min_score:.2f}", color=line_color, va='bottom', ha='right')
                        else:
                            ax[idx].axhline(y=min_score, color=line_color, linestyle='--')
                            ax[idx].axhline(y=max_score, color=line_color, linestyle='--')
                            ax[idx].text(x_max_label, max_score, f"Max: {max_score:.2f}", color=line_color, va='top', ha='left')
                            ax[idx].text(x_min_label, min_score, f"Min: {min_score:.2f}", color=line_color, va='bottom', ha='right')

                if ngames == 1:
                    ax.set_xlim(0, n_timesteps)
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True, steps=[1, 2, 5, 10]))
                    ax.xaxis.set_major_formatter(ticker.EngFormatter(places=0))
                    ax.tick_params(axis='both', which='major')
                    ax.set_title(game)
                    ax.grid(True)
                    handles, labels = ax.get_legend_handles_labels()
                    cols_num = np.ceil(len(all_models)/2)
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.8)
                        spine.set_color('#333333')
                else:
                    ax[idx].set_xlim(0, n_timesteps)
                    ax[idx].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True, steps=[1, 2, 5, 10]))
                    ax[idx].xaxis.set_major_formatter(ticker.EngFormatter(places=0))
                    ax[idx].tick_params(axis='both', which='major')
                    ax[idx].set_title(game)
                    ax[idx].grid(True)
                    handles, labels = ax[0].get_legend_handles_labels()                
                    if len(all_models) > 6:
                        cols_num = np.ceil(len(all_models)/2)
                    else:
                        cols_num = len(all_models)
                    for spine in ax[idx].spines.values():
                        spine.set_linewidth(0.8)
                        spine.set_color('#333333')
                    
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=cols_num)
                
            fig.tight_layout()
            plt.show()
            f.write(f"\n-------------------------------------------------------------------------------------------------------------\n")
            
# Create the parser
parser = argparse.ArgumentParser(description="plots")

# Add arguments
parser.add_argument("--n_models", type=int, default=4, help="number of models")
parser.add_argument("--min_max", type=int, default=1, help="add min and max lines")
parser.add_argument('--ndirs_atari', nargs='+')
parser.add_argument('--ndirs_cartpole', nargs='+')
parser.add_argument('--ndirs_mountaincar', nargs='+')

# Parse the arguments
args = parser.parse_args()
n_models = args.n_models
base_folders_atari = args.ndirs_atari
base_folders_cartpole = args.ndirs_cartpole
base_folders_mountaincar = args.ndirs_mountaincar
base_folders_other = args.ndirs_other
add_min_max = True if args.min_max == 1 else False

base_folders = {
    'Atari': base_folders_atari,
    'CartPole': base_folders_cartpole,
    'MountainCar': base_folders_mountaincar,
}

all_games = {
    'Atari': ['Asterix', 'Breakout', 'Freeway', 'Seaquest', 'SpaceInvaders'], 
    'CartPole': ['CartPole'], 
    'MountainCar': ['MountainCar'], 
}
n_steps = {
    'Atari': [2500000], 
    'CartPole': [5000], 
    'MountainCar': [100000],
}
subsample = {
    'Atari': [1000], 
    'CartPole': [2], 
    'MountainCar': [40], 
}
rolling_window = {
    'Atari': [100000], 
    'CartPole': [200], 
    'MountainCar': [4000], 
}

plot_folder = "/home/sr8685/ijcai25/eval_runs/unc_aware_buffer_plots/plots/" + datetime.now().strftime('%Y-%m-%d-%H%M%S')

for k in all_games.keys():
    games = all_games[k]
    n_timesteps = n_steps[k]
    sample = subsample[k]
    rolling_window_size = rolling_window[k]
    ndirs = base_folders[k]
    if ndirs is not None:
        ngames = len(games)
        log_folder = plot_folder + "_results/" + str(k)
        os.makedirs(log_folder)
        print(f"\nLog Folder: {log_folder}")
        plots(n_models, n_timesteps[0], rolling_window_size[0], sample[0], ndirs, ngames, games, add_min_max, log_folder)
        plot_file = log_folder + "/results_" + k + "_minmax_" + str(add_min_max) + '.png'
        plt.savefig(plot_file, format='png', bbox_inches='tight', dpi=300)
        print(f"\nFile_name: {plot_file}\n")
        