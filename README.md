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
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 64 --headless \
    +log_root_path="/home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results" \
    +exp_name="vlm_franka_train"
``` 

#### Off-Screen Recording Training
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 64 \
    --headless \
    --video \
    +log_root_path="/home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results" \
    +exp_name="vlm_franka_video_train"
```

#### Visual Training 
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 64 \
    +log_root_path="/home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results" \
    +exp_name="vlm_franka_train"
```

#### Check RL Training Results in Tensorboard:
```bash
tensorboard --logdir /home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results
``` 

#### To check the training results:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 16 \
    --checkpoint /home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results/vlm_franka_train/model.zip
``` 



