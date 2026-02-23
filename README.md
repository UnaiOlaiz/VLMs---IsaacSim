### Start BentoML VLM Service Command:

```bash
bentoml serve server_bento:VLMServiceIsaac --port 8000 --reload
```

### Launch RL Training with **IsaacLab**:
```bash
cd ~/Documents/IsaacLab
``` 
#### Headlessly
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 64 --headless log_root_path=~/Documents/PFG/Scripts/RL/results exp_name=vlm_franka_train
``` 

#### Off-Screen Recording Training
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64 --headless --video
```

#### Visual Training 
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Lift-Cube-Franka-v
0 --num_envs 64 log_root_path=~/Documents/PFG/Scripts/RL/results exp_name=vlm_fra
nka_train
```

#### Check RL Training Results in Tensorboard:
```bash
tensorboard --logdir ~/Documents/PFG/Scripts/RL/results 
``` 

#### To check the training results:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 32 --use_last_checkpoint



