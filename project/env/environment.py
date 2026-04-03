"""
Custom Gymnasium-compatible visual navigation environment.

The agent must navigate a 10x10 grid to reach a target while avoiding obstacles.
The state is returned as an 84x84 RGB image.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from env.renderer import GridRenderer


class VisualNavigationEnv(gym.Env):
    """
    A 2D visual navigation environment compatible with Gymnasium.
    
    The agent navigates a grid world to reach a target while avoiding obstacles.
    Observations are RGB images rendered from the grid state.
    
    Attributes:
        grid_size: Size of the grid (default 10x10)
        render_size: Size of rendered images (default 84x84)
        max_steps: Maximum steps per episode (default 100)
        num_obstacles: Number of obstacles to place (default 5)
    """
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}
    
    # Action definitions
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    
    # Action to direction mapping (row_delta, col_delta)
    ACTION_TO_DIRECTION = {
        ACTION_UP: (-1, 0),
        ACTION_DOWN: (1, 0),
        ACTION_LEFT: (0, -1),
        ACTION_RIGHT: (0, 1),
    }
    
    def __init__(
        self,
        grid_size: int = 10,
        render_size: int = 84,
        max_steps: int = 100,
        num_obstacles: int = 5,
        render_mode: Optional[str] = "rgb_array",
        seed: Optional[int] = None
    ):
        """
        Initialize the environment.
        
        Args:
            grid_size: Number of cells in each dimension
            render_size: Output image size in pixels
            max_steps: Maximum steps before episode terminates
            num_obstacles: Number of obstacles to place
            render_mode: Rendering mode ('rgb_array' or 'human')
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.render_size = render_size
        self.max_steps = max_steps
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        
        # Initialize renderer
        self.renderer = GridRenderer(grid_size=grid_size, render_size=render_size)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(render_size, render_size, 3),
            dtype=np.uint8
        )
        
        # State variables (will be set in reset)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.target_pos: Tuple[int, int] = (0, 0)
        self.obstacle_positions: List[Tuple[int, int]] = []
        self.current_step: int = 0
        
        # Set random seed if provided
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        else:
            self._np_random = np.random.RandomState()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional random seed
            options: Optional configuration options
            
        Returns:
            observation: Initial RGB image observation
            info: Additional information dictionary
        """
        # Handle seeding
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        
        self.current_step = 0
        
        # Generate all positions
        all_positions = [
            (r, c) for r in range(self.grid_size) for c in range(self.grid_size)
        ]
        
        # Randomly select positions for agent, target, and obstacles
        # Ensure they don't overlap
        selected_indices = self._np_random.choice(
            len(all_positions),
            size=2 + self.num_obstacles,
            replace=False
        )
        
        selected_positions = [all_positions[i] for i in selected_indices]
        
        self.agent_pos = selected_positions[0]
        self.target_pos = selected_positions[1]
        self.obstacle_positions = selected_positions[2:]
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "agent_pos": self.agent_pos,
            "target_pos": self.target_pos,
            "distance_to_target": self._manhattan_distance(self.agent_pos, self.target_pos)
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            observation: New RGB image observation
            reward: Reward for this step
            terminated: Whether episode ended (goal or collision)
            truncated: Whether episode was cut short (max steps)
            info: Additional information dictionary
        """
        self.current_step += 1
        
        # Calculate new position
        direction = self.ACTION_TO_DIRECTION[action]
        new_row = self.agent_pos[0] + direction[0]
        new_col = self.agent_pos[1] + direction[1]
        
        # Check if new position is valid (within bounds)
        if self._is_valid_position(new_row, new_col):
            self.agent_pos = (new_row, new_col)
        
        # Calculate reward and check termination
        reward = -0.1  # Step penalty
        terminated = False
        truncated = False
        success = False
        
        # Check if agent reached target
        if self.agent_pos == self.target_pos:
            reward = 10.0
            terminated = True
            success = True
        
        # Check if agent hit obstacle
        elif self.agent_pos in self.obstacle_positions:
            reward = -10.0
            terminated = True
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
        
        observation = self._get_observation()
        
        info = {
            "agent_pos": self.agent_pos,
            "target_pos": self.target_pos,
            "distance_to_target": self._manhattan_distance(self.agent_pos, self.target_pos),
            "success": success,
            "steps": self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self.render_mode == "rgb_array":
            return self._get_observation()
        return None
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation as an RGB image.
        
        Returns:
            numpy array of shape (render_size, render_size, 3)
        """
        return self.renderer.render(
            agent_pos=self.agent_pos,
            target_pos=self.target_pos,
            obstacle_positions=self.obstacle_positions
        )
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """
        Check if a position is within grid bounds.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def _manhattan_distance(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
            
        Returns:
            Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def close(self) -> None:
        """Clean up resources."""
        pass


class FeatureObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that transforms image observations to ViT feature embeddings.
    
    This wrapper is used for the feature-based RL pipeline, converting
    84x84 RGB images into 768-dimensional ViT embeddings.
    """
    
    def __init__(self, env: gym.Env, vit_encoder, feature_dim: int = 768):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            vit_encoder: ViTEncoder instance for feature extraction
            feature_dim: Dimension of the output features
        """
        super().__init__(env)
        
        self.vit_encoder = vit_encoder
        self.feature_dim = feature_dim
        
        # Update observation space to match feature dimension
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim,),
            dtype=np.float32
        )
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Transform image observation to feature embedding.
        
        Args:
            observation: RGB image array (H, W, 3)
            
        Returns:
            Feature embedding array (feature_dim,)
        """
        # Extract features using ViT encoder
        features = self.vit_encoder.encode(observation)
        return features.astype(np.float32)
