# 🐍 Snake DQN Agent

A Snake game where the player is a **deep Q-learning agent**: it starts out clueless, crashes into walls, and slowly learns to hunt food.

Built as my final project for the *Intelligent Agents* course (University of Toronto, SCS, 2024).

## What's inside

| Piece | What it does |
|---|---|
| **Deep Q-Network** | A 3-layer dense network (256 → 256 → 128) that looks at the 16×16 grid and scores the 4 possible moves. |
| **Target network** | A frozen copy of the model, synced every 4 episodes, so training doesn't chase its own tail. |
| **Experience replay** | A 5,000-move memory; the agent trains on random batches of 64 past moves instead of only the latest one. |
| **ε-greedy exploration** | Starts 85% random and decays to 12.5%, so it explores early and exploits later. |
| **Reward shaping** | +5000 for food, −1000 for hitting a wall or its own tail. |
| **"Kryptonite" memory** | Cells where the snake died get flagged as dangerous, and random moves avoid them. |
| **No-reverse rule** | The snake can't turn 180° into itself. |
| **Levels** | Each food eaten levels up and speeds the game up (8 → 22 FPS). |

## Run it

```bash
pip install -r requirements.txt
python snake_dqn.py
```

It trains live in a pygame window for up to one hour (or 25,000 episodes). Close the window to stop.

## What I learned

- **Reward design is everything.** Early versions rewarded the wrong things and the snake learned to spin in circles.
- **A raw grid is a hard input.** The agent learns slowly from 256 pixels; hand-built features (danger ahead or left or right, food direction) would learn much faster. That's my next step.
- **Training per step is slow.** `model.fit` on every sample is simple but expensive; batching the replay update is a clear speed-up.

## Tech
Python · TensorFlow/Keras · NumPy · pygame
