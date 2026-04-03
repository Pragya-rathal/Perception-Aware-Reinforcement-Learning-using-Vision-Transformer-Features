"""
Main entry point for the Perception-Aware RL project.

This script orchestrates the complete experimental pipeline:
1. Train pixel-based RL model
2. Train feature-based RL model (using ViT)
3. Compare model performance
4. Generate visualization plots
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perception-Aware RL: Compare Pixel vs ViT Feature-Based Learning"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps per model (default: 100000)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    
    parser.add_argument(
        "--skip-pixel",
        action="store_true",
        help="Skip pixel-based training"
    )
    
    parser.add_argument(
        "--skip-feature",
        action="store_true",
        help="Skip feature-based training"
    )
    
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip comparison and plotting"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid size for environment (default: 10)"
    )
    
    parser.add_argument(
        "--num-obstacles",
        type=int,
        default=5,
        help="Number of obstacles (default: 5)"
    )
    
    return parser.parse_args()


def main():
    """Run the complete experimental pipeline."""
    args = parse_args()
    
    # Print banner
    print("\n" + "=" * 70)
    print("PERCEPTION-AWARE REINFORCEMENT LEARNING")
    print("Comparing Pixel-Based vs ViT Feature-Based RL")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {args.results_dir}")
    print(f"Total timesteps per model: {args.timesteps:,}")
    print(f"Random seed: {args.seed}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Number of obstacles: {args.num_obstacles}")
    print("=" * 70 + "\n")
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment configuration
    env_kwargs = {
        "grid_size": args.grid_size,
        "num_obstacles": args.num_obstacles,
        "seed": args.seed,
    }
    
    # Storage for metrics
    pixel_metrics = None
    feature_metrics = None
    
    # ========== Phase 1: Train Pixel-Based Model ==========
    if not args.skip_pixel:
        print("\n" + "=" * 70)
        print("PHASE 1: Training Pixel-Based Model")
        print("=" * 70 + "\n")
        
        from rl.train_pixel import PixelTrainer
        
        pixel_trainer = PixelTrainer(
            env_kwargs=env_kwargs,
            save_dir=args.results_dir,
            verbose=1
        )
        
        pixel_metrics = pixel_trainer.train(
            total_timesteps=args.timesteps,
            eval_freq=max(args.timesteps // 20, 1000),
            n_eval_episodes=10
        )
        
        print("\nPixel-Based Training Complete!")
        print(f"  Final avg reward (last 100): {sum(pixel_metrics['episode_rewards'][-100:])/100:.2f}")
        print(f"  Final success rate (last 100): {sum(pixel_metrics['episode_successes'][-100:])/100*100:.1f}%")
    
    # ========== Phase 2: Train Feature-Based Model ==========
    if not args.skip_feature:
        print("\n" + "=" * 70)
        print("PHASE 2: Training Feature-Based Model (ViT)")
        print("=" * 70 + "\n")
        
        from rl.train_feature import FeatureTrainer
        
        feature_trainer = FeatureTrainer(
            env_kwargs=env_kwargs,
            save_dir=args.results_dir,
            verbose=1
        )
        
        feature_metrics = feature_trainer.train(
            total_timesteps=args.timesteps,
            eval_freq=max(args.timesteps // 20, 1000),
            n_eval_episodes=10
        )
        
        print("\nFeature-Based Training Complete!")
        print(f"  Final avg reward (last 100): {sum(feature_metrics['episode_rewards'][-100:])/100:.2f}")
        print(f"  Final success rate (last 100): {sum(feature_metrics['episode_successes'][-100:])/100*100:.1f}%")
    
    # ========== Phase 3: Compare Models ==========
    if not args.skip_compare:
        print("\n" + "=" * 70)
        print("PHASE 3: Model Comparison")
        print("=" * 70 + "\n")
        
        from experiments.compare import ModelComparator
        
        comparator = ModelComparator(results_dir=args.results_dir)
        
        comparison_results = comparator.compare(
            pixel_metrics=pixel_metrics,
            feature_metrics=feature_metrics
        )
        
        comparator.print_comparison(comparison_results)
        comparator.save_comparison(comparison_results)
        
        # ========== Phase 4: Generate Plots ==========
        if not args.no_plots:
            print("\n" + "=" * 70)
            print("PHASE 4: Generating Visualizations")
            print("=" * 70 + "\n")
            
            from experiments.plots import ResultsPlotter
            
            plotter = ResultsPlotter(results_dir=args.results_dir)
            
            plotter.plot_all(
                pixel_metrics=pixel_metrics,
                feature_metrics=feature_metrics,
                comparison_stats=comparison_results.comparison_stats,
                show=False  # Don't display, just save
            )
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {results_dir.absolute()}")
    print("\nFiles generated:")
    
    for f in sorted(results_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  • {f.name} ({size_str})")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
