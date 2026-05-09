# Integration of Vision Language Models (VLMs) in a Multi-Agent environment for cooperative robotic tasks inside NVIDIA Isaac Sim

## Description

This project proposes the development and implementation of a Vision Language Model integrated inside a Multi-Agent scenario for the execution of cooperative robotic tasks within a simulated ``NVIDIA's Isaac Sim`` environment.

<p align="center">
    <img src="Documentation_LaTeX/imgs/Multi-agent_scenario.png" alt="Multi-agent cooperative robotic scenario">
</p>

## Table of Contents

- [Project Structure](#project-structure-details)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Open Source License](#open-source-license)

## Project Structure Details

- **Documentation_LaTeX/**: Full project documentation and thesis
- **Models/**: Pre-trained RL policies and model checkpoints
- **Scenarios/**: Isaac Sim environment configurations and USD scene files
- **Scripts/**: Implementation code organized by module:
  - `CameraSim/`: Camera simulation and visual detection
  - `Control/`: Robot controller implementations
  - `RL/`: Reinforcement learning training and policy testing
  - `Demos/`: Jupyter notebooks with demonstrations
  - `Server/`: BentoML server configuration
- **media/**: Results, VLM outputs, and experimental data
- **Notes/**: Documentation and progress tracking
- **logs/**: Training logs and tensorboard data

## Features

- **VLM integration**: High-level decision-making orchestrators using pretrained SOTA models such as ``Qwen2-VL``.
- **Multi-agent cooperative environments**: Multiple robotic agents will have to cooperate to achieve a global task in simulated environments. 
- **IsaacSim as physics simulator**: Will be using NVIDIA's SOTA physics simulator to carry on all the experiments in a controlled environment. 
- **Custom workflow pipeline**: VLM integration custom pipeline ``REST API``-based, including communication betweeen the local simulation and a remote inference server. 
- **BentoML server**: Inference will be held in a ``BentoML`` server hosting the VLM. 
- **Reinforcement learning**: Policy training experimentation using ``NVIDIA's Isaac Lab`` with advanced RL frameworks such as ``Stable Baselines 3 (SB3)``. 
- **Computer vision algorithms**: CV filters to enhance the VLM's detecting performance.
- **Tensorboard** for following RL training progressions visually.

## System Architecture

The project is divided into two main components:

### Local Component (Isaac Sim)
- **Physics simulation and rendering** using ``NVIDIA Omniverse``. 
- **Camera capture** and computer vision processing.
- **Robot controllers** for multi-agent coordination. 

### Remote Component (BentoML Server)
- **Vision Language Model inference** using Qwen2-VL.
- **High-level decision making** based on visual input and natural language reasoning.
- **Modular and scalabile**
- **VRAM and resource optimization** through BentoML.

### Communication Bridge
- **REST API** for seamless communication between local and remote components.
- **HTTP-based request/response protocol** with ``JSON`` data format.

## Installation

### Prerequisites (recommended)
Even if it is possible to run the project with lower-fidelity hardware, NVIDIA's official suggestion is the following: 

| Element | Minimum Spec | Good Spec | Ideal Spec |
|---------|-------------|-----------|-----------|
| **OS** | Ubuntu 22.04/24.04<br/>Windows 10/11 | Ubuntu 22.04/24.04<br/>Windows 10/11 | Ubuntu 22.04/24.04<br/>Windows 10/11 |
| **CPU** | Intel Core i7 (7th Gen)<br/>AMD Ryzen 5 | Intel Core i7 (9th Gen)<br/>AMD Ryzen 7 | Intel Core i9, X-series or higher<br/>AMD Ryzen 9, Threadripper or higher |
| **Cores** | 4 | 8 | 16 |
| **RAM** | 32GB | 64GB | 64GB |
| **Storage** | 50GB SSD | 500GB SSD | 1TB NVMe SSD |
| **GPU** | GeForce RTX 4080 | GeForce RTX 5080 | RTX PRO 6000 Blackwell |
| **VRAM** | 16GB | 16GB | 48GB |
| **Driver** | Linux: 580.65.06<br/>Windows: 580.88 | Linux: 580.65.06<br/>Windows: 580.88 | Linux: 580.65.06<br/>Windows: 580.88 |

### Setup Steps

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/UnaiOlaiz/VLMs---IsaacSim.git
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API tokens. The worked has been carried out using open-source models which often ask for a custom ``API key`` necessary for trying them out.
   - Add your NVIDIA VLM API key to `tokens/api_key_nvidia.txt`
   - Add your HuggingFace token to `tokens/hugging_token.txt`

4. Set up IsaacLab environment. Also recommended to clone the official ``Isaac Lab`` repository. 
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   ```

## Usage

### Start BentoML VLM Service

```bash
cd Scripts/Server
bentoml serve server_bento:VLMServiceIsaac --port 8000 --reload
```

### Launch IsaacSim (installed via pip)
```bash 
isaacsim
```

### Launch RL Training with Isaac Lab

#### Headless Training (faster, no visualization)
```bash
cd ~/Documents/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task <task_name> --num_envs <num_envs> --headless
```

#### Visual Training (lower <num_envs> for rendering)
```bash
cd ~/Documents/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task <task_name> --num_envs <num_envs>
```

<p align="center">
    <img src="media/Franka levanta cubo.gif" alt="Lift cube GIF">
</p>

#### Monitor Training with TensorBoard
```bash
tensorboard --logdir ~/Documents/IsaacLab/logs/sb3/<task_name>
```

#### Test Trained Models
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task <task_name> --num_envs <num_envs> --use_last_checkpoint
```

## Open Source License

This project is totally  open-source and free to use!











