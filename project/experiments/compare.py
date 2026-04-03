"""
Model comparison utilities for evaluating pixel vs. feature-based RL.

Provides functions to compare training curves, convergence speed,
and final performance between different RL approaches.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ComparisonResults:
    """Container for comparison results between models."""
    
    pixel_metrics: Dict[str, Any]
    feature_metrics: Dict[str, Any]
    comparison_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pixel_metrics": self.pixel_metrics,
            "feature_metrics": self.feature_metrics,
            "comparison_stats": self.comparison_stats,
        }


class ModelComparator:
    """
    Compares performance between pixel-based and feature-based RL models.
    
    Analyzes training curves, convergence speed, success rates,
    and statistical significance of differences.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the comparator.
        
        Args:
            results_dir: Directory containing training results
        """
        self.results_dir = Path(results_dir)
    
    def load_metrics(self, metrics_file: str) -> Dict[str, Any]:
        """
        Load metrics from a JSON file.
        
        Args:
            metrics_file: Path to metrics JSON file
            
        Returns:
            Metrics dictionary
        """
        with open(metrics_file, "r") as f:
            return json.load(f)
    
    def compute_rolling_stats(
        self,
        values: List[float],
        window: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute rolling mean and standard deviation.
        
        Args:
            values: List of values
            window: Rolling window size
            
        Returns:
            Tuple of (rolling_mean, rolling_std)
        """
        arr = np.array(values)
        n = len(arr)
        
        if n < window:
            return arr, np.zeros_like(arr)
        
        rolling_mean = np.convolve(arr, np.ones(window)/window, mode='valid')
        
        # Compute rolling std
        rolling_std = np.array([
            np.std(arr[max(0, i-window+1):i+1])
            for i in range(window-1, n)
        ])
        
        return rolling_mean, rolling_std
    
    def find_convergence_episode(
        self,
        rewards: List[float],
        threshold: float = 0.9,
        success_reward: float = 10.0,
        window: int = 100
    ) -> int:
        """
        Find the episode at which the model converges.
        
        Convergence is defined as the first episode where the rolling
        mean reward exceeds threshold * success_reward.
        
        Args:
            rewards: List of episode rewards
            threshold: Fraction of max reward to consider converged
            success_reward: Maximum reward for success
            window: Window size for rolling mean
            
        Returns:
            Episode number at convergence, or -1 if not converged
        """
        target = threshold * success_reward
        rolling_mean, _ = self.compute_rolling_stats(rewards, window)
        
        for i, mean in enumerate(rolling_mean):
            if mean >= target:
                return i + window  # Account for window offset
        
        return -1  # Not converged
    
    def compute_sample_efficiency(
        self,
        rewards: List[float],
        target_reward: float = 5.0,
        window: int = 100
    ) -> int:
        """
        Compute sample efficiency as episodes to reach target reward.
        
        Args:
            rewards: List of episode rewards
            target_reward: Target average reward
            window: Window size for rolling mean
            
        Returns:
            Number of episodes to reach target, or -1 if not reached
        """
        rolling_mean, _ = self.compute_rolling_stats(rewards, window)
        
        for i, mean in enumerate(rolling_mean):
            if mean >= target_reward:
                return i + window
        
        return -1
    
    def compare(
        self,
        pixel_metrics_file: Optional[str] = None,
        feature_metrics_file: Optional[str] = None,
        pixel_metrics: Optional[Dict[str, Any]] = None,
        feature_metrics: Optional[Dict[str, Any]] = None,
    ) -> ComparisonResults:
        """
        Compare pixel-based and feature-based models.
        
        Args:
            pixel_metrics_file: Path to pixel model metrics
            feature_metrics_file: Path to feature model metrics
            pixel_metrics: Direct metrics dict for pixel model
            feature_metrics: Direct metrics dict for feature model
            
        Returns:
            ComparisonResults object
        """
        # Load metrics from files if not provided directly
        if pixel_metrics is None:
            if pixel_metrics_file is None:
                pixel_metrics_file = self.results_dir / "pixel_metrics.json"
            pixel_metrics = self.load_metrics(pixel_metrics_file)
        
        if feature_metrics is None:
            if feature_metrics_file is None:
                feature_metrics_file = self.results_dir / "feature_metrics.json"
            feature_metrics = self.load_metrics(feature_metrics_file)
        
        # Extract rewards and successes
        pixel_rewards = pixel_metrics["episode_rewards"]
        feature_rewards = feature_metrics["episode_rewards"]
        pixel_successes = pixel_metrics["episode_successes"]
        feature_successes = feature_metrics["episode_successes"]
        
        # Compute rolling statistics
        pixel_rolling_mean, pixel_rolling_std = self.compute_rolling_stats(pixel_rewards)
        feature_rolling_mean, feature_rolling_std = self.compute_rolling_stats(feature_rewards)
        
        # Final performance (last 100 episodes)
        pixel_final_reward = np.mean(pixel_rewards[-100:])
        feature_final_reward = np.mean(feature_rewards[-100:])
        pixel_final_success = np.mean(pixel_successes[-100:])
        feature_final_success = np.mean(feature_successes[-100:])
        
        # Convergence speed
        pixel_convergence = self.find_convergence_episode(pixel_rewards)
        feature_convergence = self.find_convergence_episode(feature_rewards)
        
        # Sample efficiency
        pixel_efficiency = self.compute_sample_efficiency(pixel_rewards)
        feature_efficiency = self.compute_sample_efficiency(feature_rewards)
        
        # Training time
        pixel_time = pixel_metrics.get("training_time_seconds", 0)
        feature_time = feature_metrics.get("training_time_seconds", 0)
        
        # Compile comparison statistics
        comparison_stats = {
            "pixel": {
                "final_mean_reward": float(pixel_final_reward),
                "final_success_rate": float(pixel_final_success),
                "convergence_episode": pixel_convergence,
                "sample_efficiency_episode": pixel_efficiency,
                "total_episodes": len(pixel_rewards),
                "training_time_seconds": pixel_time,
                "max_rolling_reward": float(np.max(pixel_rolling_mean)) if len(pixel_rolling_mean) > 0 else 0,
            },
            "feature": {
                "final_mean_reward": float(feature_final_reward),
                "final_success_rate": float(feature_final_success),
                "convergence_episode": feature_convergence,
                "sample_efficiency_episode": feature_efficiency,
                "total_episodes": len(feature_rewards),
                "training_time_seconds": feature_time,
                "max_rolling_reward": float(np.max(feature_rolling_mean)) if len(feature_rolling_mean) > 0 else 0,
            },
            "comparison": {
                "reward_improvement": float(feature_final_reward - pixel_final_reward),
                "success_rate_improvement": float(feature_final_success - pixel_final_success),
                "convergence_speedup": (
                    pixel_convergence - feature_convergence
                    if pixel_convergence > 0 and feature_convergence > 0
                    else None
                ),
                "time_difference_seconds": feature_time - pixel_time,
            }
        }
        
        return ComparisonResults(
            pixel_metrics=pixel_metrics,
            feature_metrics=feature_metrics,
            comparison_stats=comparison_stats,
        )
    
    def print_comparison(self, results: ComparisonResults) -> None:
        """
        Print a formatted comparison report.
        
        Args:
            results: ComparisonResults object
        """
        stats = results.comparison_stats
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON RESULTS")
        print("=" * 70)
        
        # Table header
        print(f"\n{'Metric':<35} {'Pixel-Based':>15} {'Feature-Based':>15}")
        print("-" * 70)
        
        # Metrics
        print(f"{'Final Mean Reward':<35} "
              f"{stats['pixel']['final_mean_reward']:>15.2f} "
              f"{stats['feature']['final_mean_reward']:>15.2f}")
        
        print(f"{'Final Success Rate':<35} "
              f"{stats['pixel']['final_success_rate']*100:>14.1f}% "
              f"{stats['feature']['final_success_rate']*100:>14.1f}%")
        
        print(f"{'Max Rolling Reward':<35} "
              f"{stats['pixel']['max_rolling_reward']:>15.2f} "
              f"{stats['feature']['max_rolling_reward']:>15.2f}")
        
        print(f"{'Convergence Episode':<35} "
              f"{stats['pixel']['convergence_episode']:>15} "
              f"{stats['feature']['convergence_episode']:>15}")
        
        print(f"{'Sample Efficiency (episodes)':<35} "
              f"{stats['pixel']['sample_efficiency_episode']:>15} "
              f"{stats['feature']['sample_efficiency_episode']:>15}")
        
        print(f"{'Total Episodes':<35} "
              f"{stats['pixel']['total_episodes']:>15} "
              f"{stats['feature']['total_episodes']:>15}")
        
        print(f"{'Training Time (seconds)':<35} "
              f"{stats['pixel']['training_time_seconds']:>15.1f} "
              f"{stats['feature']['training_time_seconds']:>15.1f}")
        
        print("-" * 70)
        
        # Summary
        print("\nSUMMARY:")
        comp = stats["comparison"]
        
        if comp["reward_improvement"] > 0:
            print(f"  • Feature-based model achieves {comp['reward_improvement']:.2f} higher average reward")
        else:
            print(f"  • Pixel-based model achieves {-comp['reward_improvement']:.2f} higher average reward")
        
        if comp["success_rate_improvement"] > 0:
            print(f"  • Feature-based model has {comp['success_rate_improvement']*100:.1f}% higher success rate")
        else:
            print(f"  • Pixel-based model has {-comp['success_rate_improvement']*100:.1f}% higher success rate")
        
        if comp["convergence_speedup"] is not None:
            if comp["convergence_speedup"] > 0:
                print(f"  • Feature-based model converges {comp['convergence_speedup']} episodes faster")
            else:
                print(f"  • Pixel-based model converges {-comp['convergence_speedup']} episodes faster")
        
        print("=" * 70 + "\n")
    
    def save_comparison(
        self,
        results: ComparisonResults,
        output_file: Optional[str] = None
    ) -> None:
        """
        Save comparison results to JSON.
        
        Args:
            results: ComparisonResults object
            output_file: Output file path
        """
        if output_file is None:
            output_file = self.results_dir / "comparison_results.json"
        
        with open(output_file, "w") as f:
            json.dump(results.comparison_stats, f, indent=2)
        
        print(f"Comparison results saved to {output_file}")


def compare_models(
    results_dir: str = "results",
    pixel_metrics: Optional[Dict[str, Any]] = None,
    feature_metrics: Optional[Dict[str, Any]] = None,
) -> ComparisonResults:
    """
    Convenience function to compare models.
    
    Args:
        results_dir: Directory containing results
        pixel_metrics: Optional direct metrics for pixel model
        feature_metrics: Optional direct metrics for feature model
        
    Returns:
        ComparisonResults object
    """
    comparator = ModelComparator(results_dir=results_dir)
    results = comparator.compare(
        pixel_metrics=pixel_metrics,
        feature_metrics=feature_metrics
    )
    comparator.print_comparison(results)
    comparator.save_comparison(results)
    return results


if __name__ == "__main__":
    # Run comparison on existing results
    results = compare_models()
