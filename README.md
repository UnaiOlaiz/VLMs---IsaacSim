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
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Lift-Cube-Franka-IK-Abs-VLM-v0 --num_envs 4096 --headless
``` 

#### Visual Training 
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Lift-Cube-Franka-IK-Abs-VLM-v0 --num_envs 64
```

#### Check RL Training Results in Tensorboard:
```bash
tensorboard --logdir ~/Documents/IsaacLab/logs/sb3/Isaac-Lift-Cube-Franka-IK-Abs-VLM-v0
``` 

#### To check the training results:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Lift-Cube-Franka-IK-Abs-VLM-v0 --num_envs 16 --use_last_checkpoint
``` 
Or with a specific checkpoint: 
```bash
find logs -type f | grep "Isaac-Lift-Cube-Franka-IK-Abs-VLM-v0" | grep ".zip"
```
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py \
    --task Isaac-Lift-Cube-Franka-IK-Abs-VLM-v0 \
    --num_envs 1 \
    --checkpoint logs/sb3/Isaac-Lift-Cube-Franka-IK-Abs-VLM-v0/2026-03-08_12-10-12/model_40960000_steps.zip
```



