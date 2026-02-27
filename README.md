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
cd ~/Documents/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 64 --headless
``` 

#### Off-Screen Recording Training
```bash
# Cleanest version for your README
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 64 \
    --headless \
    --video
```

#### Visual Training 
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 64
```

#### Check RL Training Results in Tensorboard:
```bash
tensorboard --logdir ~/Documents/IsaacLab/logs/sb3/Isaac-Lift-Cube-Franka-v0
``` 

#### To check the training results:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 16 --use_last_checkpoint
``` 



