#!/usr/bin/env python3
"""
Minimal evaluation script for RxR CMA baseline.
No training dependencies - just model + environment.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from habitat import Config as HabitatConfig
from habitat import make_dataset
from habitat.config.default import get_config as get_habitat_config
from habitat_baselines.common.environments import get_env_class
from tqdm import tqdm

from config import Config
from models.cma_policy import CMAPolicy


def setup_habitat_config(args):
    """Create Habitat config for RxR dataset"""
    config = get_habitat_config()
    
    config.defrost()
    
    # Dataset
    config.DATASET.TYPE = "RxR-VLN-CE-v1"
    config.DATASET.SPLIT = args.split
    config.DATASET.DATA_PATH = f"{args.dataset}/{{split}}/{{split}}_{{role}}.json.gz"
    config.DATASET.SCENES_DIR = args.scenes
    
    # Task
    config.TASK.TYPE = "VLN-v0"
    config.TASK.SENSORS = ["RXR_INSTRUCTION_SENSOR"]
    config.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL"]
    config.TASK.SUCCESS.SUCCESS_DISTANCE = 3.0
    
    # RxR Instruction Sensor
    config.TASK.RXR_INSTRUCTION_SENSOR = HabitatConfig()
    config.TASK.RXR_INSTRUCTION_SENSOR.TYPE = "RxRInstructionSensor"
    config.TASK.RXR_INSTRUCTION_SENSOR.features_path = (
        f"{args.text_features}/rxr_{{split}}/{{id:06}}_{{lang}}_text_features.npz"
    )
    
    # Simulator
    config.SIMULATOR.TYPE = "Sim-v0"
    config.SIMULATOR.ACTION_SPACE_CONFIG = "v1"
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    config.SIMULATOR.TURN_ANGLE = 30
    config.SIMULATOR.TILT_ANGLE = 30
    
    config.SIMULATOR.RGB_SENSOR.WIDTH = 640
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 480
    config.SIMULATOR.RGB_SENSOR.HFOV = 79
    
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480
    config.SIMULATOR.DEPTH_SENSOR.HFOV = 79
    config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.5
    config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.AGENT_0.HEIGHT = 0.88
    config.SIMULATOR.AGENT_0.RADIUS = 0.18
    
    # Environment
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 500
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    
    config.freeze()
    return config


def load_model(checkpoint_path, observation_space, action_space, device):
    """Load pretrained CMA model"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Create model config
    model_config = Config()
    
    # Create model
    model = CMAPolicy(
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config,
    )
    
    # Load weights
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def evaluate(model, env, num_episodes, device):
    """Run evaluation on episodes"""
    
    results = defaultdict(list)
    episode_count = 0
    max_episodes = num_episodes if num_episodes > 0 else len(env.episodes)
    
    print(f"Starting evaluation on {max_episodes} episodes...")
    
    with tqdm(total=max_episodes) as pbar:
        while episode_count < max_episodes:
            observations = env.reset()
            episode_id = env.current_episode.episode_id
            
            # Initialize RNN state
            rnn_states = torch.zeros(
                1, model.net.num_recurrent_layers, 512, device=device
            )
            prev_actions = torch.zeros(1, 1, dtype=torch.long, device=device)
            not_done_masks = torch.ones(1, 1, dtype=torch.uint8, device=device)
            
            # Store trajectory
            trajectory = []
            trajectory.append({
                "position": env.sim.get_agent_state().position.tolist(),
                "rotation": env.sim.get_agent_state().rotation.components.tolist(),
            })
            
            done = False
            while not done:
                # Prepare batch
                batch = {
                    "rgb": torch.from_numpy(observations["rgb"]).unsqueeze(0).to(device),
                    "depth": torch.from_numpy(observations["depth"]).unsqueeze(0).to(device),
                    "rxr_instruction": torch.from_numpy(observations["rxr_instruction"]).unsqueeze(0).to(device),
                }
                
                # Get action from model
                with torch.no_grad():
                    action, rnn_states = model.act(
                        batch, rnn_states, prev_actions, not_done_masks, deterministic=True
                    )
                
                prev_actions.copy_(action)
                
                # Step environment
                observations, reward, done, info = env.step(action.item())
                
                # Store position
                trajectory.append({
                    "position": env.sim.get_agent_state().position.tolist(),
                    "rotation": env.sim.get_agent_state().rotation.components.tolist(),
                    "action": action.item(),
                })
            
            # Store episode results
            results[episode_id] = {
                "trajectory": trajectory,
                "distance_to_goal": info.get("distance_to_goal", -1),
                "success": info.get("success", 0),
                "spl": info.get("spl", 0),
            }
            
            episode_count += 1
            pbar.update(1)
            pbar.set_postfix({
                "success": info.get("success", 0),
                "dtg": f"{info.get('distance_to_goal', -1):.2f}",
            })
    
    return results


def compute_metrics(results):
    """Compute aggregate metrics"""
    metrics = {
        "num_episodes": len(results),
        "success_rate": np.mean([r["success"] for r in results.values()]),
        "spl": np.mean([r["spl"] for r in results.values()]),
        "distance_to_goal": np.mean([r["distance_to_goal"] for r in results.values()]),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to RxR dataset")
    parser.add_argument("--scenes", required=True, help="Path to Matterport3D scenes")
    parser.add_argument("--text-features", required=True, help="Path to BERT text features")
    parser.add_argument("--split", default="val_unseen", help="Dataset split")
    parser.add_argument("--num-episodes", type=int, default=-1, help="Number of episodes (-1 for all)")
    parser.add_argument("--output", default="results.json", help="Output file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Habitat config
    habitat_config = setup_habitat_config(args)
    
    # Create environment
    print("Creating environment...")
    env_class = get_env_class("Env")
    env = env_class(config=habitat_config)
    
    print(f"Loaded {len(env.episodes)} episodes from {args.split}")
    
    # Load model
    model = load_model(
        args.checkpoint,
        env.observation_space,
        env.action_space,
        device
    )
    
    # Run evaluation
    results = evaluate(model, env, args.num_episodes, device)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {metrics['num_episodes']}")
    print(f"Success Rate: {metrics['success_rate']:.3f}")
    print(f"SPL: {metrics['spl']:.3f}")
    print(f"Distance to Goal: {metrics['distance_to_goal']:.2f}m")
    print("="*50)
    
    # Save results
    output = {
        "metrics": metrics,
        "episodes": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()