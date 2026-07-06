import os
from dataclasses import dataclass, field

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    min_max: bool = True
    """if toggled, min and max values will be plotted in the plot"""
    notes: str = "plots"
    """notes regarding the run"""
    
    # Algorithm specific arguments
    category: str = 'Atari'
    """category of games [Atari/Other(Classic Control, Toy Text)]"""
    games: list[str] = field(default_factory=lambda: ['Asterix', 'Breakout', 'Freeway', 'Seaquest', 'SpaceInvaders'])
    """names of the game environments"""
    logdirs: list[str] | None = None
    """the log directories for the experiments"""
    legend: list[str] | None = None
    """name of the atari plot labels (legend)"""
    smooth: int = 11
    """averaging window size for smoothing the plot curves"""
    grid_res: int = 1000
    """grid resolution for the shared uniform grid"""
    n_timesteps: int = 300000
    """timesteps for the game environment"""
    n_models: int = 4
    """the number of models that are evaluated"""
    sample: int = 1000
    """sampling window size"""
    rolling_window: int = 1000
    """rolling window size"""