import os
import sys
import tyro
import torch
import random
import gymnasium as gym
from pathlib import Path
from toy_text_args import Args
from toy_text_udrm import UDRM
import wandb
from dataclasses import asdict

# 1. Get the current directory
current_dir = Path(__file__).resolve().parent

# 2. Get the parent directory
parent_dir = current_dir.parent

# 3. Add parent directory to sys.path if it isn't already there
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from log_utils import run_time, calculate_stats
from models import MLP_Toy_UDRM

# Create the parser
args = tyro.cli(Args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps = [args.total_timesteps for i in range(len(args.env_ids))]

op_fldrs = []
avg_cum_rewards = []
max_scores = []
cum_reward = []
game_count = []
run_seeds = []

i = args.game_idx
env_id = args.env_ids[i]
total_run_time = run_time()

if args.seed is None:
    args.seed = []
else:
    args.seed = list(args.seed)

for sd in range(args.n_seeds):
    run_time_calc = run_time()
    curr_seed = args.seed[sd] if sd < len(args.seed) else random.randint(0, 2e8)
    run_seeds.append(curr_seed)

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=env_id + args.wandb_group,
        name=f"{env_id}{args.wandb_group}_seed_{curr_seed}",
        config=asdict(args)
    )
    run_url = wandb.run.url

    if env_id == "CliffWalking-v0":
        env = gym.make(env_id, max_episode_steps=1000)
        env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    currentfolder = args.log_folders[i]
    obs_space_ch = args.obs_space_chs[i]
    unc_lambda_ep = args.unc_lambdas_ep[i]
    unc_lambda_al = args.unc_lambdas_al[i]
    num_steps = steps[i]
    evi_coeff = args.evi_coeffs[i]
    agent = UDRM(
                env,
                MLP_Toy_UDRM,
                n_quantiles = args.n_quantiles,
                timesteps = num_steps,
                obs_space_channel = obs_space_ch,
                kappa = args.kappa,
                replay_start_size = args.learning_starts,
                replay_buffer_size = args.buffer_size,
                alpha_int = args.alpha,
                beta_int = args.beta,
                gamma = args.gamma,
                tau = args.tau,
                # update_target_frequency = args.update_target_frequency,
                batch_size = args.batch_size,
                learning_rate = args.learning_rate,
                adam_epsilon = args.adam_epsilon,
                update_frequency = args.update_frequency,
                evi_coeff = evi_coeff,
                unc_lambda_ep = unc_lambda_ep,
                unc_lambda_al = unc_lambda_al,
                log_folder_details = currentfolder,
                train_file = os.path.abspath(__file__),
                wandb_url = run_url,
                seed = curr_seed,
                logging = args.logging,
                save_period = args.save_period,
                notes = args.notes
            )

    print(f"Output Folder: {agent.logger.train_details['output_folder']}\n", flush=True)
    cum_score, max_score = agent.learn(verbose=True)
    curr_avg = round(sum(cum_score)/len(cum_score), 4)
    avg_cum_rewards.append(curr_avg)
    curr_max = max(max_score)
    curr_count = len(max_score)
    curr_cum = round(sum(cum_score), 4)
    max_scores.append(curr_max)
    game_count.append(curr_count)
    cum_reward.append(round(curr_cum, 4))
    op_fldrs.append(agent.logger.log_folder)
    agent.save()
    
    with open(agent.logger.log_folder + "/log_output.txt", "a") as f:
        n = len(agent.logger.log_folder) + 15
        dashes = "-" * n
        print(f"\nGame: {env_id}")
        f.write("\nGame: " + env_id)
        print(f"Seed: {agent.seed}")
        f.write(f"\nSeed: {agent.seed}")
        print(f"\nTimesteps: {num_steps}")
        f.write(f"\nTimesteps: {num_steps}")
        print(f"Buffer size: {agent.replay_buffer_size}")
        f.write(f"\nBuffer size: {agent.replay_buffer_size}")
        print(f"Replay start: {agent.replay_start_size}")
        f.write(f"\nReplay start: {agent.replay_start_size}")
        print(f"\nMax score: {curr_max}")
        f.write(f"\n\nMax score: {curr_max}")
        print(f"Cum reward: {curr_cum}")
        f.write(f"\nCum reward: {curr_cum}")
        cnt = len(max_score)
        print(f"Total games: {cnt}")
        f.write(f"\nTotal games: {cnt}")
        print(f"Avg cum reward: {curr_avg}")
        f.write(f"\nAvg cum reward: {curr_avg}")
        mean_cum_score = sum(cum_score)/cnt if cnt > 0 else -1
        print(f"\n{dashes}\nFiles saved at {agent.logger.log_folder}\n{dashes}")
        f.write(f"\n{dashes}\nFiles saved at {agent.logger.log_folder}\n{dashes}")
    
        run_time_calc.block_end()
        run_time_calc.print_time(n)
        f.write(f"\n{dashes}\nStart time: {run_time_calc.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        f.write(f"\nEnd time: {run_time_calc.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        f.write(f"\n\nTotal execution time:")
        f.write(f"\n{run_time_calc.days} days, {run_time_calc.hours} hours, {run_time_calc.minutes} minutes, {run_time_calc.seconds} seconds, and {run_time_calc.microseconds} microseconds\n{dashes}")
        print(f"\nExecuted file: {os.path.abspath(__file__)}")
        f.write(f"\nExecuted file: {os.path.abspath(__file__)}")
        f.write(f"\nWandb URL: {run_url}\n")
        f.write(f"\n{calculate_stats([curr_max], metric='max_scores')}")
        f.write(f"\n{calculate_stats([curr_avg], metric='avg_cum_rewards')}")
    f.close()
    print(f"Wandb URL: {run_url}\n")
    wandb.finish()

root_file = agent.logger.root_folder + total_run_time.start_time.strftime('%Y-%m-%d-%H%M%S_') + env_id + "_info.txt"
game_name = agent.logger.log_folder_details.split("-")[1]
filedir = os.path.realpath(__file__)
folder_idx = filedir.rfind("/")
folder_dir = filedir[:folder_idx+1]
root_file = agent.logger.root_folder + total_run_time.start_time.strftime('%Y-%m-%d-%H%M%S_Toy_Text-') + game_name + "_info.txt"
total_run_time.block_end()
with open(root_file, "w") as f:
    f.write(f"\nGame: {env_id}")
    f.write(f"\nMax episode steps: {env.spec.max_episode_steps}")
    f.write(f"\nTimesteps: {agent.train_steps}")
    f.write(f"\nBuffer size: {agent.replay_buffer_size}")
    f.write(f"\nReplay start: {agent.replay_start_size}\n")
    f.write(f"\nSeeds: {run_seeds}")
    f.write(f"\nMax score: {max_scores}")
    f.write(f"\nCum score: {cum_reward}")
    f.write(f"\nTotal episodes: {game_count}")
    f.write(f"\nCum reward: {avg_cum_rewards}\n")
    f.write(f"\n{calculate_stats(max_scores, metric='max_scores')}")
    f.write(f"\n{calculate_stats(avg_cum_rewards, metric='avg_cum_rewards')}")
    f.write(f"\n\nOutput Folders:\n")
    f.write("\n".join(str(item) for item in op_fldrs))
    f.write(f"\n\n{dashes}\nStart time: {total_run_time.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    f.write(f"\nEnd time: {total_run_time.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    f.write(f"\n\nTotal execution time:")
    f.write(f"\n{total_run_time.days} days, {total_run_time.hours} hours, {total_run_time.minutes} minutes, {total_run_time.seconds} seconds, and {total_run_time.microseconds} microseconds\n{dashes}");
    f.write(f"\nExecuted file: {os.path.abspath(__file__)}");
f.close()
print(f"\nSummary written to: {root_file}\n")