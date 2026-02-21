Start BentoML VLM Service Command:

```bash
bentoml serve server_bento:VLMServiceIsaac --port 8000 --reload
```

Launch RL Training:
```bash
cd ~/Documents/PFG/Scripts/RL/results
python3 ../franka_rl.py --num_envs <num_envs>
``` 

Check RL Training Results in Tensorboard:

```bash
tensorboard --logdir=/home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results/tensorboard_franka/
``` 
