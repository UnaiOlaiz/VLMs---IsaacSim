Start BentoML VLM Service Command:

```bash
bentoml serve server_bento:VLMServiceIsaac --port 8000 --reload
```

Launch RL Training:
```bash
cd ~/Documents/PFG/Scripts/RL/results
/home/unaiolaizolaosa/isaac-sim-5.1.0/python.sh ../franka_rl.py --num_envs <num_envs>
``` 

Check RL Training Results in Tensorboard:

```bash
tensorboard --logdir=/home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results/tensorboard_franka/
``` 

To check (visually or not) the training results:
```bash
xhost +local:docker

docker run --name isaac-sim-pfg --entrypoint bash -it --runtime=nvidia --gpus all \
  --user root \
  -e "ACCEPT_EULA=Y" --rm --network=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/unaiolaizolaosa/Documents/PFG:/project \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  nvcr.io/nvidia/isaac-sim:5.1.0

cd /project/Scripts/RL
/isaac-sim/python.sh test_train.py
``` 



