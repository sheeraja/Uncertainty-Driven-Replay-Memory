import os
from dataclasses import dataclass, field

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    notes: str = "CEQRDQN"
    """notes regarding the run"""
    wandb_entity: str = "exploruder"
    """the entity (team) of wandb's project"""
    wandb_project: str = "UDRM"
    """the wandb's project name"""
    wandb_group: str = "_UDRM"
    """group for related experiment seeds"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_ids: list[str] = field(default_factory=lambda: ["FrozenLake-v1", "CliffWalking-v0"])
    """the name of the game environment (default: CartPole-v1)"""
    log_folders: list[str] = field(default_factory=lambda: ["Gym-FrozenLake-CEQRDQN", "Gym-CliffWalking-CEQRDQN"])
    """log folder names"""
    game_idx: int = 0
    """the index of the game environment"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    buffer_size: int = 50000
    """the replay memory buffer size"""
    learning_starts: int = 10000
    """timestep to start learning"""
    seed: list[int] | None = None
    """seed of the experiment"""
    n_seeds: int = 10
    """number of seeds to run the experiments for"""
    tau: float = 0.001
    """smooth update rate (Polyak averaging) for the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    n_quantiles: int = 20
    """the number of quantiles for quantile regression"""
    obs_space_chs: list[int] = field(default_factory=lambda: [16, 48])
    """number of channels in the observation space"""
    kappa: int = 1
    """maximum value for calculating the huber loss"""
    gamma: float = 0.99
    """the discount factor gamma"""
    update_target_frequency: int = 50
    """the frequency to update the target network"""
    learning_rate: float = 6e-3
    """the learning rate of the optimizer"""
    adam_epsilon: float = 1e-8
    """epsilon for Adam optimizer"""
    update_frequency: int = 1
    """the frequency to update the main network"""
    evi_coeffs: list[float] = field(default_factory=lambda: [0.5, 0.5])
    """evidential coefficient for the regularization loss"""
    unc_lambdas_ep: list[float] = field(default_factory=lambda: [0.001, 0.001])
    """lambda for epistemic uncertainty for Thompson sampling"""
    unc_lambdas_al: list[float] = field(default_factory=lambda: [0, 0])
    """lambda for aleatoric uncertainty for Risk-averse"""
    logging: bool = True
    """whether to log metrics"""
    save_period: int = 5000
    """the period to save the model (in timesteps)"""