# Moska with Monte Carlo Tree Search
This repository contains an implementation of a Monte Carlo Tree Search (MCTS)-based AI agent for a Finnish hidden-information card game Moska. The project includes the game engine, player interfaces and tools to benchmark and develop MCTS-based agents.

The code in this repository is part of a thesis studying MCTS in Moska, which can found [here](https://urn.fi/URN:NBN:fi-fe2025082083496).

## Overview
Moska is a hidden information trick-taking card game popular in Eastern Finland. The game is played with a standard 52-card deck and the objective of the game is to not be the last player holding cards.

The imperfect information environment of Moska presents unique challenges for AI agents, as they must make decisions based on incomplete knowledge. Monte Carlo Tree Search was applied to the hidden information environment using a technique called game state determinization. This technique involves creating a deterministic version of the game state by making assumptions about the hidden information, allowing the MCTS algorithm to explore possible future game states. 

Two different determinization strategies were studied: 
- Random determinization: unknown cards are assigned randomly
- Neural network-based determinization: a neural network is used to predict the most likely cards held by each player

## Project structure
```
moska/
├── moskaengine/      # Core game engine and players
├── research/         # Neural network training, game state data creation
├── benchmarks/       # Benchmarking and evaluation of different configurations
├── models/           # Pre-trained models for card prediction
├── sandbox.py        # Testing script for the project
├── requirements.txt  # Dependencies
├── setup.py          # Setup script
├── pyproject.toml    # Build system config
└── README.md         # Documentation
```

## Installation and requirements
Tested on Python 3.12.

1. Clone the repository:
    ```bash
    git clone https://github.com/arnotoro/moska
    cd moska
    ```
2. Install the dependencies:
    ```bash
    pip install -e .
    ```
    or with requirements:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Play a game in terminal:
```bash
python sandbox.py
```
To change the player configurations, edit the `sandbox.py` players list.

## Known bugs
- Playing a game with two MCTS players can lead to the game crashing or hanging indefinitely.

<!-- ## Future improvements -->

## License
This project is licensed under the MIT License.

The game engine and MCTS implementation for Moska is a heavily modified version from a similar [Durak](https://github.com/jorisperrenet/durak) implementation, which is a related card game to Moska due to the similarities in gameplay. Inspiration for the project was also drawn from previous research on Moska by a fellow student and a good friend of mine. His work can be found [here](https://github.com/ilmari99/MoskaResearch). 