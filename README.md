# Ultimo-MARL

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Environment](https://img.shields.io/badge/Environment-PettingZoo%20%2F%20Gymnasium-brightgreen.svg)](https://pettingzoo.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

A Multi-Agent Reinforcement Learning (MARL) environment and agent implementation in Python and PyTorch for the multiplayer card game Ultimo. Models gameplay as an imperfect information game with strict fog-of-war constraints.

---

## Features

- **PettingZoo AEC Environment:** Observation vector (~70 features) encoding visible table state, personal hand, and opponent card counts without information leaks.
- **Dueling DQN with PER:** Self-play training pipeline with action masking to prevent illegal moves and prioritized experience replay.
- **Live TCP Bot:** Standalone client connecting directly over raw TCP sockets to the authoritative Ultimo server to play live games.

---

## Getting Started

### Prerequisites
- Python 3.11+
- PyTorch 2.0+

### Installation & Running
```bash
git clone https://github.com/StanislawGolec/Ultimo-MARL.git
cd Ultimo-MARL
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Run live bot against local Ultimo server
python run_live_bot.py --host localhost --port 8080 --name Bot_DQN
```
