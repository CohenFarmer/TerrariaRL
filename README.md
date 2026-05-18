# TerrariaRL

Training an RL agent to beat Terraria from scratch using pure intrinsic motivation, no game-specific knowledge, no reward shaping, no human demonstrations.

Stage 1 is to see whether the agent can even learn to chop wood, mine ores through intrinisic curiosity, two networks are compared, a fixed network (never trained), and the predictor network. 
The error between the two networks is the reward.

## Project Structure

```
TerrariaRL/
├── mod/                    # tModLoader C# mod (runs inside Terraria)
│   ├── TerrariaRL.cs       # Mod entry point
│   ├── RLBridge.cs         # TCP server, game state extraction
│   ├── RLPlayer.cs         # Input injection via ProcessTriggers
│   └── build.txt           # Mod metadata
│
├── env/                    # Python Gymnasium environment
│   ├── terraria_env.py     # Gym wrapper (step, reset, obs/action spaces)
│   └── test_connection.py  # Manual test client
│
├── agent/                  # RL agent and training
│   ├── curiosity/          # RND / intrinsic motivation modules
│   ├── world_model/        # Learned dynamics model (Dreamer-style)
│   └── train.py            # Training entrypoint
│
├── configs/                # Hyperparameters, experiment configs
├── logs/                   # Training logs, tensorboard
├── docs/                   # Notes, architecture diagrams, findings
├── .gitignore
└── README.md
```

## Status

- [x] tModLoader mod
- [x] Action injection via ProcessTriggers
- [x] Python test client, randome movement
- [ ] Gymnasium environment wrapper
- [ ] RND curiosity module
- [ ] World model
- [ ] Training pipeline
- [ ] First exploration results
