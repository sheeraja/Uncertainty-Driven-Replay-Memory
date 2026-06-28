import os
from dataclasses import dataclass, field

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    notes: str = "UADQN"
    """notes regarding the run"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_ids: list[str] = field(default_factory=lambda: ['asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders'])
    """the names of the game environments (default: Asterix-v1)"""
    log_folders: list[str] = field(default_factory=lambda: ["MinAtar-Asterix-UADQN","MinAtar-Breakout-UADQN","MinAtar-Freeway-UADQN","MinAtar-Seaquest-UADQN","MinAtar-SpaceInvaders-UADQN"])
    """log folder names"""
    game_idx: int = 0
    """the index of the game environment"""
    total_timesteps: int = 2500000
    """total timesteps of the experiments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    learning_starts: int = 5000
    """timestep to start learning"""
    seed: list[int] | None = None
    """seed of the experiment"""
    seed_beg: int = 0
    """beginning seed index"""
    seed_end: int = 10
    """ending seed index"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    n_quantiles: int = 50
    """the number of quantiles for quantile regression"""
    obs_space_chs: list[int] = field(default_factory=lambda: [4, 4, 7, 10, 6])
    """number of channels in the observation space"""
    kappa: int = 0
    """maximum value for calculating the huber loss"""
    gamma: float = 0.99
    """the discount factor gamma"""
    update_target_frequency: int = 1000
    """the frequency to update the target network"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    adam_epsilon: float = 1e-8
    """epsilon for Adam optimizer"""
    update_frequency: int = 1
    """the frequency to update the main network"""
    evi_coeffs: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5, 0.5])
    """evidential coefficient for the regularization loss"""
    unc_lambdas_ep: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2])
    """lambda for epistemic uncertainty for Thompson sampling"""
    unc_lambdas_al: list[float] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    """lambda for aleatoric uncertainty for Risk-averse"""
    weight_scale: list[float] = field(default_factory=lambda: [3, 3, 3, 3, 3])  
    """the scale of the weight noise for CNNMinAtar"""
    noise_scale: list[float] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    """the scale of the noise for anchor loss"""
    logging: bool = True
    """whether to log metrics"""
    save_period: int = 1e7
    """the period to save the model (in timesteps)"""