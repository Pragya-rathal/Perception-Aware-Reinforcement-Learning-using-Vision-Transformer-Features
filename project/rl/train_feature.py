"""
Feature-based RL training pipeline.

Trains a PPO agent on ViT feature embeddings instead of raw pixels,
using an MLP policy for faster learning.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env.environment import VisualNavigationEnv
from models.vit_encoder import ViTEncoder


class ViTFeatureWrapper(gym.ObservationWrapper):
    """
    Gym wrapper that transforms image observations to ViT features.
    
    Maintains a reference to a shared ViT encoder to avoid
    loading multiple copies of the model.
    """
    
    def __init__(self, env: gym.Env, vit_encoder: ViTEncoder):
        """
        Initialize the wrapper.
        
        Args:
            env: Environment to wrap
            vit_encoder: Shared ViT encoder instance
        """
        super().__init__(env)
        
        self.vit_encoder = vit_encoder
        self.feature_dim = vit_encoder.get_embedding_dim()
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_dim,),
            dtype=np.float32
        )
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Transform image to ViT features."""
        features = self.vit_encoder.encode(observation)
        return features.astype(np.float32)


class FeatureReducer(BaseFeaturesExtractor):
    """
    Custom feature extractor that reduces ViT features to a smaller dimension.
    
    Takes 768-dim ViT embeddings and projects them to 128-dim
    for more efficient policy learning.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128
    ):
        """
        Initialize the feature reducer.
        
        Args:
            observation_space: Input observation space
            features_dim: Output feature dimension
        """
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        self.reducer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the reducer network."""
        return self.reducer(observations)


class FeatureTrainingCallback(BaseCallback):
    """
    Callback to track training metrics for feature-based RL.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_successes: List[bool] = []
        self.current_episode_reward: float = 0.0
        self.current_episode_length: int = 0
    
    def _on_step(self) -> bool:
        """Called after each step."""
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            info = self.locals["infos"][0]
            success = info.get("success", False)
            self.episode_successes.append(success)
            
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
            
            if self.verbose > 0 and len(self.episode_rewards) % 100 == 0:
                recent_rewards = self.episode_rewards[-100:]
                recent_successes = self.episode_successes[-100:]
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Avg Reward = {np.mean(recent_rewards):.2f}, "
                      f"Success Rate = {np.mean(recent_successes)*100:.1f}%")
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return collected metrics."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
        }


class FeatureTrainer:
    """
    Trainer for feature-based RL using ViT embeddings and MLP policy.
    
    Uses a pretrained ViT to extract features from images, then trains
    a simple MLP policy on these features.
    """
    
    def __init__(
        self,
        env_kwargs: Dict[str, Any] = None,
        model_kwargs: Dict[str, Any] = None,
        vit_model_name: str = None,
        feature_reducer_dim: int = 128,
        save_dir: str = "results",
        verbose: int = 1
    ):
        """
        Initialize the trainer.
        
        Args:
            env_kwargs: Arguments for environment creation
            model_kwargs: Arguments for PPO model creation
            vit_model_name: HuggingFace model name for ViT
            feature_reducer_dim: Dimension to reduce features to
            save_dir: Directory to save results
            verbose: Verbosity level
        """
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.vit_model_name = vit_model_name
        self.feature_reducer_dim = feature_reducer_dim
        self.save_dir = Path(save_dir)
        self.verbose = verbose
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ViT encoder (shared across all environments)
        print("Initializing ViT encoder...")
        self.vit_encoder = ViTEncoder(model_name=vit_model_name)
        
        # Initialize components
        self.env = None
        self.model = None
        self.metrics_callback = None
    
    def create_env(self) -> gym.Env:
        """Create and wrap the environment with ViT feature extraction."""
        def make_env():
            env = VisualNavigationEnv(**self.env_kwargs)
            env = Monitor(env)
            env = ViTFeatureWrapper(env, self.vit_encoder)
            return env
        
        env = DummyVecEnv([make_env])
        return env
    
    def create_model(self, env: gym.Env) -> PPO:
        """
        Create PPO model with MLP policy and feature reducer.
        
        Args:
            env: The training environment
            
        Returns:
            Configured PPO model
        """
        # Policy kwargs with custom feature extractor
        policy_kwargs = {
            "features_extractor_class": FeatureReducer,
            "features_extractor_kwargs": {"features_dim": self.feature_reducer_dim},
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        }
        
        default_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": self.verbose,
            "policy_kwargs": policy_kwargs,
            "tensorboard_log": str(self.save_dir / "tensorboard_feature"),
        }
        
        # Override with user-provided kwargs
        default_kwargs.update(self.model_kwargs)
        
        return PPO(**default_kwargs)
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Train the feature-based agent.
        
        Args:
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            n_eval_episodes: Episodes per evaluation
            
        Returns:
            Training metrics dictionary
        """
        print("=" * 60)
        print("Starting Feature-Based RL Training (ViT)")
        print("=" * 60)
        
        # Create environment and model
        self.env = self.create_env()
        self.model = self.create_model(self.env)
        
        # Setup callbacks
        self.metrics_callback = FeatureTrainingCallback(verbose=self.verbose)
        
        # Create evaluation environment
        eval_env = self.create_env()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.save_dir / "feature_best"),
            log_path=str(self.save_dir / "feature_eval"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=self.verbose
        )
        
        # Train
        start_time = datetime.now()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[self.metrics_callback, eval_callback],
            progress_bar=True
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save final model
        model_path = self.save_dir / "feature_model_final"
        self.model.save(str(model_path))
        print(f"Model saved to {model_path}")
        
        # Collect metrics
        metrics = self.metrics_callback.get_metrics()
        metrics["training_time_seconds"] = training_time
        metrics["total_timesteps"] = total_timesteps
        metrics["model_type"] = "feature"
        
        # Save metrics
        metrics_path = self.save_dir / "feature_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "episode_rewards": metrics["episode_rewards"],
                "episode_successes": [bool(s) for s in metrics["episode_successes"]],
                "training_time_seconds": training_time,
                "total_timesteps": total_timesteps,
            }, f, indent=2)
        
        print(f"\nTraining completed in {training_time:.1f} seconds")
        print(f"Total episodes: {len(metrics['episode_rewards'])}")
        
        # Cleanup
        self.env.close()
        eval_env.close()
        
        return metrics
    
    def evaluate(
        self,
        model_path: Optional[str] = None,
        n_episodes: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to saved model (uses current model if None)
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        if model_path is not None:
            env = self.create_env()
            model = PPO.load(model_path, env=env)
        else:
            env = self.env
            model = self.model
        
        rewards = []
        successes = []
        lengths = []
        
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for _ in range(n_episodes * 100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if done[0]:
                rewards.append(episode_reward)
                lengths.append(episode_length)
                successes.append(info[0].get("success", False))
                
                episode_reward = 0
                episode_length = 0
                
                if len(rewards) >= n_episodes:
                    break
        
        metrics = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": np.mean(successes),
            "mean_length": np.mean(lengths),
        }
        
        if model_path is not None:
            env.close()
        
        return metrics


def train_feature_agent(
    total_timesteps: int = 100000,
    save_dir: str = "results",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train a feature-based agent.
    
    Args:
        total_timesteps: Total training steps
        save_dir: Directory to save results
        **kwargs: Additional arguments for FeatureTrainer
        
    Returns:
        Training metrics
    """
    trainer = FeatureTrainer(save_dir=save_dir, **kwargs)
    return trainer.train(total_timesteps=total_timesteps)


if __name__ == "__main__":
    # Run standalone training
    metrics = train_feature_agent(total_timesteps=50000)
    
    print("\nFinal Results:")
    print(f"Episodes trained: {len(metrics['episode_rewards'])}")
    print(f"Final avg reward (last 100): {np.mean(metrics['episode_rewards'][-100:]):.2f}")
    print(f"Final success rate (last 100): {np.mean(metrics['episode_successes'][-100:])*100:.1f}%")
