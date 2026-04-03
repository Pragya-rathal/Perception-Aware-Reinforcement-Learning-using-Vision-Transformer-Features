"""
Pixel-based RL training pipeline.

Trains a PPO agent directly on raw 84x84 RGB images using
Stable-Baselines3's CNN policy.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.environment import VisualNavigationEnv


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to track training metrics.
    
    Records episode rewards and success rates during training
    for later analysis and comparison.
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
        # Accumulate rewards
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # Check for episode end
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Check success from info
            info = self.locals["infos"][0]
            success = info.get("success", False)
            self.episode_successes.append(success)
            
            # Reset counters
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


class PixelTrainer:
    """
    Trainer for pixel-based RL using CNN policy.
    
    Uses PPO with a CNN feature extractor to learn directly
    from raw image observations.
    """
    
    def __init__(
        self,
        env_kwargs: Dict[str, Any] = None,
        model_kwargs: Dict[str, Any] = None,
        save_dir: str = "results",
        verbose: int = 1
    ):
        """
        Initialize the trainer.
        
        Args:
            env_kwargs: Arguments for environment creation
            model_kwargs: Arguments for PPO model creation
            save_dir: Directory to save results
            verbose: Verbosity level
        """
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.save_dir = Path(save_dir)
        self.verbose = verbose
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.env = None
        self.model = None
        self.metrics_callback = None
    
    def create_env(self) -> gym.Env:
        """Create and wrap the environment."""
        def make_env():
            env = VisualNavigationEnv(**self.env_kwargs)
            env = Monitor(env)
            return env
        
        # Create vectorized environment
        env = DummyVecEnv([make_env])
        
        # Transpose images for CNN (channels first)
        env = VecTransposeImage(env)
        
        return env
    
    def create_model(self, env: gym.Env) -> PPO:
        """
        Create PPO model with CNN policy.
        
        Args:
            env: The training environment
            
        Returns:
            Configured PPO model
        """
        default_kwargs = {
            "policy": "CnnPolicy",
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
            "tensorboard_log": str(self.save_dir / "tensorboard_pixel"),
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
        Train the pixel-based agent.
        
        Args:
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            n_eval_episodes: Episodes per evaluation
            
        Returns:
            Training metrics dictionary
        """
        print("=" * 60)
        print("Starting Pixel-Based RL Training")
        print("=" * 60)
        
        # Create environment and model
        self.env = self.create_env()
        self.model = self.create_model(self.env)
        
        # Setup callbacks
        self.metrics_callback = TrainingMetricsCallback(verbose=self.verbose)
        
        # Create evaluation environment
        eval_env = self.create_env()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.save_dir / "pixel_best"),
            log_path=str(self.save_dir / "pixel_eval"),
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
        model_path = self.save_dir / "pixel_model_final"
        self.model.save(str(model_path))
        print(f"Model saved to {model_path}")
        
        # Collect metrics
        metrics = self.metrics_callback.get_metrics()
        metrics["training_time_seconds"] = training_time
        metrics["total_timesteps"] = total_timesteps
        metrics["model_type"] = "pixel"
        
        # Save metrics
        metrics_path = self.save_dir / "pixel_metrics.json"
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
        # Load model if path provided
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
        
        for _ in range(n_episodes * 100):  # Enough steps for n_episodes
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


def train_pixel_agent(
    total_timesteps: int = 100000,
    save_dir: str = "results",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train a pixel-based agent.
    
    Args:
        total_timesteps: Total training steps
        save_dir: Directory to save results
        **kwargs: Additional arguments for PixelTrainer
        
    Returns:
        Training metrics
    """
    trainer = PixelTrainer(save_dir=save_dir, **kwargs)
    return trainer.train(total_timesteps=total_timesteps)


if __name__ == "__main__":
    # Run standalone training
    metrics = train_pixel_agent(total_timesteps=50000)
    
    print("\nFinal Results:")
    print(f"Episodes trained: {len(metrics['episode_rewards'])}")
    print(f"Final avg reward (last 100): {np.mean(metrics['episode_rewards'][-100:]):.2f}")
    print(f"Final success rate (last 100): {np.mean(metrics['episode_successes'][-100:])*100:.1f}%")
