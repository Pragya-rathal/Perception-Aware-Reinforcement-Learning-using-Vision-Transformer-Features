"""
Visualization functions for RL training results.

Generates plots comparing pixel-based and feature-based RL training,
including reward curves, success rates, and convergence analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


class ResultsPlotter:
    """
    Creates visualizations for comparing RL training results.
    """
    
    # Color scheme
    COLORS = {
        "pixel": "#2E86AB",      # Blue
        "feature": "#A23B72",    # Magenta
        "pixel_fill": "#2E86AB", 
        "feature_fill": "#A23B72",
    }
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the plotter.
        
        Args:
            results_dir: Directory containing results and for saving plots
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_metrics(self, metrics_file: str) -> Dict[str, Any]:
        """Load metrics from JSON file."""
        with open(metrics_file, "r") as f:
            return json.load(f)
    
    def compute_rolling_stats(
        self,
        values: List[float],
        window: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute rolling mean, std, and episode indices.
        
        Returns:
            Tuple of (episodes, rolling_mean, rolling_std)
        """
        arr = np.array(values)
        n = len(arr)
        
        if n < window:
            episodes = np.arange(n)
            return episodes, arr, np.zeros_like(arr)
        
        rolling_mean = np.convolve(arr, np.ones(window)/window, mode='valid')
        
        rolling_std = np.array([
            np.std(arr[max(0, i-window+1):i+1])
            for i in range(window-1, n)
        ])
        
        episodes = np.arange(window-1, n)
        
        return episodes, rolling_mean, rolling_std
    
    def plot_reward_curves(
        self,
        pixel_rewards: List[float],
        feature_rewards: List[float],
        window: int = 100,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot reward curves comparing both models.
        
        Args:
            pixel_rewards: Episode rewards for pixel model
            feature_rewards: Episode rewards for feature model
            window: Rolling window size
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Compute rolling statistics
        pixel_eps, pixel_mean, pixel_std = self.compute_rolling_stats(pixel_rewards, window)
        feature_eps, feature_mean, feature_std = self.compute_rolling_stats(feature_rewards, window)
        
        # Plot pixel-based rewards
        ax.plot(pixel_eps, pixel_mean, 
                color=self.COLORS["pixel"], 
                label="Pixel-Based (CNN)",
                linewidth=2)
        ax.fill_between(pixel_eps, 
                        pixel_mean - pixel_std, 
                        pixel_mean + pixel_std,
                        color=self.COLORS["pixel_fill"], 
                        alpha=0.2)
        
        # Plot feature-based rewards
        ax.plot(feature_eps, feature_mean, 
                color=self.COLORS["feature"], 
                label="Feature-Based (ViT)",
                linewidth=2)
        ax.fill_between(feature_eps, 
                        feature_mean - feature_std, 
                        feature_mean + feature_std,
                        color=self.COLORS["feature_fill"], 
                        alpha=0.2)
        
        # Reference lines
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Max Reward')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Labels and legend
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Reward (Rolling Mean, window={window})")
        ax.set_title("Training Reward Comparison: Pixel vs Feature-Based RL")
        ax.legend(loc='lower right')
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved reward curves to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_success_rates(
        self,
        pixel_successes: List[bool],
        feature_successes: List[bool],
        window: int = 100,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot success rate comparison.
        
        Args:
            pixel_successes: Success flags for pixel model
            feature_successes: Success flags for feature model
            window: Rolling window size
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to float for rolling computation
        pixel_success_float = [float(s) for s in pixel_successes]
        feature_success_float = [float(s) for s in feature_successes]
        
        # Compute rolling success rates
        pixel_eps, pixel_rate, _ = self.compute_rolling_stats(pixel_success_float, window)
        feature_eps, feature_rate, _ = self.compute_rolling_stats(feature_success_float, window)
        
        # Convert to percentage
        pixel_rate *= 100
        feature_rate *= 100
        
        # Plot
        ax.plot(pixel_eps, pixel_rate, 
                color=self.COLORS["pixel"], 
                label="Pixel-Based (CNN)",
                linewidth=2)
        
        ax.plot(feature_eps, feature_rate, 
                color=self.COLORS["feature"], 
                label="Feature-Based (ViT)",
                linewidth=2)
        
        # Reference line
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Success')
        
        # Labels and legend
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Success Rate % (Rolling, window={window})")
        ax.set_title("Success Rate Comparison: Pixel vs Feature-Based RL")
        ax.legend(loc='lower right')
        ax.set_xlim(left=0)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved success rate plot to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_comparison_summary(
        self,
        comparison_stats: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a summary bar chart comparing key metrics.
        
        Args:
            comparison_stats: Statistics from ModelComparator
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        pixel_stats = comparison_stats["pixel"]
        feature_stats = comparison_stats["feature"]
        
        # Bar width and positions
        width = 0.35
        x = np.array([0])
        
        # Plot 1: Final Reward
        ax1 = axes[0]
        ax1.bar(x - width/2, pixel_stats["final_mean_reward"], width, 
                color=self.COLORS["pixel"], label="Pixel-Based")
        ax1.bar(x + width/2, feature_stats["final_mean_reward"], width, 
                color=self.COLORS["feature"], label="Feature-Based")
        ax1.set_ylabel("Mean Reward")
        ax1.set_title("Final Performance\n(Last 100 Episodes)")
        ax1.set_xticks([])
        ax1.legend()
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add value labels
        ax1.annotate(f'{pixel_stats["final_mean_reward"]:.2f}', 
                     xy=(x - width/2, pixel_stats["final_mean_reward"]),
                     ha='center', va='bottom', fontsize=10)
        ax1.annotate(f'{feature_stats["final_mean_reward"]:.2f}', 
                     xy=(x + width/2, feature_stats["final_mean_reward"]),
                     ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Success Rate
        ax2 = axes[1]
        ax2.bar(x - width/2, pixel_stats["final_success_rate"] * 100, width, 
                color=self.COLORS["pixel"], label="Pixel-Based")
        ax2.bar(x + width/2, feature_stats["final_success_rate"] * 100, width, 
                color=self.COLORS["feature"], label="Feature-Based")
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_title("Final Success Rate\n(Last 100 Episodes)")
        ax2.set_xticks([])
        ax2.set_ylim(0, 105)
        ax2.legend()
        
        ax2.annotate(f'{pixel_stats["final_success_rate"]*100:.1f}%', 
                     xy=(x - width/2, pixel_stats["final_success_rate"] * 100),
                     ha='center', va='bottom', fontsize=10)
        ax2.annotate(f'{feature_stats["final_success_rate"]*100:.1f}%', 
                     xy=(x + width/2, feature_stats["final_success_rate"] * 100),
                     ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Convergence Speed (if available)
        ax3 = axes[2]
        pixel_conv = pixel_stats["convergence_episode"]
        feature_conv = feature_stats["convergence_episode"]
        
        if pixel_conv > 0 or feature_conv > 0:
            # Use -1 values as "not converged" indicator
            pixel_val = pixel_conv if pixel_conv > 0 else 0
            feature_val = feature_conv if feature_conv > 0 else 0
            
            ax3.bar(x - width/2, pixel_val, width, 
                    color=self.COLORS["pixel"], label="Pixel-Based")
            ax3.bar(x + width/2, feature_val, width, 
                    color=self.COLORS["feature"], label="Feature-Based")
            ax3.set_ylabel("Episodes to Convergence")
            ax3.set_title("Convergence Speed\n(Lower is Better)")
            ax3.set_xticks([])
            ax3.legend()
            
            pixel_label = str(pixel_conv) if pixel_conv > 0 else "N/C"
            feature_label = str(feature_conv) if feature_conv > 0 else "N/C"
            ax3.annotate(pixel_label, xy=(x - width/2, pixel_val),
                         ha='center', va='bottom', fontsize=10)
            ax3.annotate(feature_label, xy=(x + width/2, feature_val),
                         ha='center', va='bottom', fontsize=10)
        else:
            ax3.text(0.5, 0.5, "Neither model\nconverged", 
                     ha='center', va='center', transform=ax3.transAxes,
                     fontsize=12)
            ax3.set_title("Convergence Speed")
        
        plt.suptitle("Model Comparison Summary", fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved comparison summary to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_all(
        self,
        pixel_metrics: Optional[Dict[str, Any]] = None,
        feature_metrics: Optional[Dict[str, Any]] = None,
        comparison_stats: Optional[Dict[str, Any]] = None,
        show: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Generate all plots.
        
        Args:
            pixel_metrics: Metrics for pixel model
            feature_metrics: Metrics for feature model
            comparison_stats: Comparison statistics
            show: Whether to display plots
            
        Returns:
            Dictionary of generated figures
        """
        # Load metrics if not provided
        if pixel_metrics is None:
            pixel_metrics = self.load_metrics(self.results_dir / "pixel_metrics.json")
        if feature_metrics is None:
            feature_metrics = self.load_metrics(self.results_dir / "feature_metrics.json")
        if comparison_stats is None:
            comp_file = self.results_dir / "comparison_results.json"
            if comp_file.exists():
                comparison_stats = self.load_metrics(comp_file)
        
        figures = {}
        
        # Reward curves
        figures["reward_curves"] = self.plot_reward_curves(
            pixel_metrics["episode_rewards"],
            feature_metrics["episode_rewards"],
            save_path=str(self.results_dir / "reward_curves.png"),
            show=show
        )
        
        # Success rates
        figures["success_rates"] = self.plot_success_rates(
            pixel_metrics["episode_successes"],
            feature_metrics["episode_successes"],
            save_path=str(self.results_dir / "success_rates.png"),
            show=show
        )
        
        # Comparison summary (if stats available)
        if comparison_stats:
            figures["comparison_summary"] = self.plot_comparison_summary(
                comparison_stats,
                save_path=str(self.results_dir / "comparison_summary.png"),
                show=show
            )
        
        print(f"\nAll plots saved to {self.results_dir}/")
        
        return figures


def plot_training_curves(
    results_dir: str = "results",
    show: bool = True
) -> Dict[str, plt.Figure]:
    """
    Convenience function to generate all plots.
    
    Args:
        results_dir: Directory containing results
        show: Whether to display plots
        
    Returns:
        Dictionary of generated figures
    """
    plotter = ResultsPlotter(results_dir=results_dir)
    return plotter.plot_all(show=show)


if __name__ == "__main__":
    # Generate plots from existing results
    plot_training_curves()
