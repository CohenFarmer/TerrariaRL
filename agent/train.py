"""
Architecture:
- Custom feature extractor combining pixels (CNN) and structured state (MLPs)
- PPO from Stable-Baselines3 for policy training
- RND (Random Network Distillation) for intrinsic curiosity reward
- Curiosity reward added on top of environment reward
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.terraria_env import TerrariaEnv

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback



class TerrariaFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        #pixel screen layer, necessary for fine grain details of things that may change turn to turn
        self.pixel_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),     #84->20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),    #20->9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),    #9->7
            nn.ReLU(),
            nn.Flatten(),
        )
        #figure out the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            pixel_flat_size = self.pixel_cnn(dummy).shape[1]
        self.pixel_fc = nn.Linear(pixel_flat_size, 256)

        #tile layer
        self.tile_embedding = nn.Embedding(1100, 16)  #don't know exact number of terratria tiles, so added buffer
        self.tile_cnn = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  #21->11
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  #11->6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
        )

        #inventory layer, embedding for similar embeddings hopefully between similaer items
        self.item_embedding = nn.Embedding(5500, 16)  #buffer again
        self.inventory_mlp = nn.Sequential(
            nn.Linear(58 * 17, 256),  
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        #same but armor slots
        self.armor_mlp = nn.Sequential(
            nn.Linear(10 * 17, 64),
            nn.ReLU(),
        )

        #heatlh, mana etc (stats)
        self.stats_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
        )

        #npcs
        self.npcs_mlp = nn.Sequential(
            nn.Linear(5 * 5, 64),
            nn.ReLU(),
        )

        #all together
        total = 256 + 128 + 128 + 64 + 32 + 64  #pixel + tile + inv + armor + stats + npc
        self.combine = nn.Sequential(
            nn.Linear(total, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # firstly normalise to between 0 and 1
        pixels = observations["pixels"].float() / 255.0
        #switch to batch, channel, height, width
        if pixels.dim() == 4 and pixels.shape[-1] == 3:
            pixels = pixels.permute(0, 3, 1, 2) 
        pixel_feat = self.pixel_cnn(pixels)
        pixel_feat = F.relu(self.pixel_fc(pixel_feat))

        #embed the tiles, then feed to cnn
        tiles = observations["tiles"].long().clamp(0, 1099)
        tile_emb = self.tile_embedding(tiles)
        tile_emb = tile_emb.permute(0, 3, 1, 2)  #switch again
        tile_feat = self.tile_cnn(tile_emb)

        #embed item ids
        inventory = observations["inventory"].long()
        item_ids = inventory[..., 0].clamp(0, 5499)
        counts = inventory[..., 1].float().unsqueeze(-1) / 999.0  #normalise
        item_emb = self.item_embedding(item_ids) 
        inv_combined = torch.cat([item_emb, counts], dim=-1)  #[B, 58, 17]
        inv_flat = inv_combined.flatten(1)
        inv_feat = self.inventory_mlp(inv_flat)

        #Armor, same as inventory but smaller
        armor = observations["armor"].long()
        armor_ids = armor[..., 0].clamp(0, 5499)
        armor_counts = armor[..., 1].float().unsqueeze(-1) / 999.0
        armor_emb = self.item_embedding(armor_ids)
        armor_combined = torch.cat([armor_emb, armor_counts], dim=-1)
        armor_flat = armor_combined.flatten(1)
        armor_feat = self.armor_mlp(armor_flat)

        #Stats
        stats_feat = self.stats_mlp(observations["stats"])

        #NPCs, flatten
        npcs_flat = observations["npcs"].flatten(1)
        npc_feat = self.npcs_mlp(npcs_flat)

        #Combine all
        combined = torch.cat([pixel_feat, tile_feat, inv_feat, armor_feat, stats_feat, npc_feat], dim=-1)
        return self.combine(combined)

# RND Curiosity
# Two networks: one frozen random target, one trained predictor.
# prediction error = curiosity reward
# Hopeful for the agent to explore, just a initial method, probably won't have any interesting behaviour

class RNDModule(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        #Fixed target network (random, never trained)
        self.target = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        #Trainable predictor (tries to match target)
        self.predictor = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        #Freeze the target
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            target_features = self.target(x)
        pred_features = self.predictor(x)
        #Prediction error per sample (mean over feature dim)
        error = ((pred_features - target_features) ** 2).mean(dim=-1)
        return error

    def loss(self, x):
        return self.forward(x).mean()


# RND INPUT BUILDER
# We just give a structured state, we don't include things like pixels or tiles as this will be very noisy
# probably not enough as it's rare to gain items early game through chance
def build_rnd_input(observations, device):
    stats = observations["stats"].to(device).float()

    inv_ids = observations["inventory"][..., 0].to(device).float() / 5500.0

    #6 stats + 58 inv = 64
    return torch.cat([stats, inv_ids], dim=-1)


# add curiosity reward on top of the environment reward, grab the buffer add curiosity reward
#then PPO trains on the buffer
class CuriosityCallback(BaseCallback):
    def __init__(self, rnd: RNDModule, device, curiosity_weight: float = 1.0, verbose=0):
        super().__init__(verbose)
        self.rnd = rnd.to(device)
        self.device = device
        self.curiosity_weight = curiosity_weight
        self.optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=1e-4)

        #running stats for normalisation of curiosity rewards
        self.reward_running_mean = 0.0
        self.reward_running_std = 1.0
        self.reward_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        #collect buffer
        rollout_buffer = self.model.rollout_buffer

        #get obs
        observations = rollout_buffer.observations  

        #convert to tensors
        obs_tensors = {
            key: torch.as_tensor(val, device=self.device).reshape(-1, *val.shape[2:])
            for key, val in observations.items()
        }

        #Build RND input
        rnd_input = build_rnd_input(obs_tensors, self.device)

        #Train predictor
        self.optimizer.zero_grad()
        loss = self.rnd.loss(rnd_input)
        loss.backward()
        self.optimizer.step()

        #Compute curiosity rewards and add to rollout
        with torch.no_grad():
            curiosity = self.rnd(rnd_input).cpu().numpy()
            
            #Update running stats
            self.reward_count += len(curiosity)
            new_mean = curiosity.mean()
            new_std = curiosity.std() + 1e-8
            
            #Exponential moving average
            alpha = 0.01
            self.reward_running_mean = (1 - alpha) * self.reward_running_mean + alpha * new_mean
            self.reward_running_std = (1 - alpha) * self.reward_running_std + alpha * new_std
            
            #normalise curiosity
            curiosity_normalized = (curiosity - self.reward_running_mean) / max(self.reward_running_std, 1e-8)
            
            #clip extreme values
            curiosity_normalized = np.clip(curiosity_normalized, -5, 5)

        # add curiosity reward to the rollout rewards
        # Reshape to match rollout structure
        curiosity_reshaped = curiosity_normalized.reshape(rollout_buffer.rewards.shape)
        rollout_buffer.rewards += self.curiosity_weight * curiosity_reshaped

        #Recompute returns/advantages with new rewards
        if hasattr(self.model, '_last_values'):
            with torch.no_grad():
                obs_for_value = {
                    key: torch.as_tensor(val[-1], device=self.device).unsqueeze(0)
                    for key, val in observations.items()
                }
                last_values = self.model.policy.predict_values(obs_for_value).cpu().numpy().flatten()
            rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=rollout_buffer.episode_starts[-1])

        #log
        if self.verbose > 0:
            self.logger.record("curiosity/raw_mean", float(curiosity.mean()))
            self.logger.record("curiosity/normalized_mean", float(curiosity_normalized.mean()))
            self.logger.record("curiosity/rnd_loss", float(loss.item()))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--curiosity_weight", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="checkpoints/ppo_rnd")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_steps", type=int, default=128,
                       help="Steps per rollout. Smaller = more frequent updates but less stable")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    print("Creating environment...")
    env = TerrariaEnv()

    policy_kwargs = dict(
        features_extractor_class=TerrariaFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    print("Creating PPO agent...")
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  #keeps exploring
        device=device,
        verbose=1,
        tensorboard_log="logs/tensorboard/",
    )

    
    print("Creating RND curiosity module...")
    rnd = RNDModule(feature_dim=256)
    curiosity_callback = CuriosityCallback(
        rnd=rnd,
        device=device,
        curiosity_weight=args.curiosity_weight,
        verbose=1,
    )

    #train
    print(f"Training for {args.timesteps} timesteps...")
    print(f"At ~15 decisions/sec, expect ~{args.timesteps/15/60:.1f} minutes of training")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=curiosity_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
