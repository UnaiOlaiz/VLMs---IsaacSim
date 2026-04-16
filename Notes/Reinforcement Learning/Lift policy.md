Got the idea to start from a closer position, so I made a control script to get the target cube coordinates and stay on top of the cube (at safe height), reducing error margin in training when starting from a much closer position. 
Had to experiment with thousands of rewards/observations + hyperparameter configurations, but the best result obtained was with this config (I will just put the important content): 

```python
# --- VLM Data Loading ---
vlm_json_path = os.path.expanduser("~/Documents/PFG/Scripts/Control/rl_start_near_cube_v2.json")

def load_vlm_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

vlm_data = load_vlm_json(vlm_json_path)

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
		# ...

        # This enables collisions so your manual Convex Decomposition is used
        self.scene.robot.spawn.collision_props = CollisionPropertiesCfg(collision_enabled=True)        
        
        # ...

        # 6. INTEGRATE VLM LOGIC
        if vlm_data:
            print(f"[VLM INFO] Loading initialization from JSON")
            jp = vlm_data["joint_positions"]
            self.scene.robot.init_state.joint_positions = {
                f"panda_joint{i+1}": jp[i] for i in range(7)
            }
            self.scene.robot.init_state.joint_positions["panda_finger_joint1"] = jp[7]
            self.scene.robot.init_state.joint_positions["panda_finger_joint2"] = jp[8]
            self.scene.robot.default_joint_pos = self.scene.robot.init_state.joint_positions

            if "cube_world_pos" in vlm_data:
                cp = vlm_data["cube_world_pos"]
                self.scene.object.init_state.pos = (-cp[0], -cp[1], 0.06)

        # Disable randomization for VLM starts
        self.events.reset_object_position = None

# more
```

```bash
seed: 42
n_timesteps: 25000000
policy: MlpPolicy
n_steps: 32
batch_size: 4096
gae_lambda: 0.95
gamma: 0.99
n_epochs: 4
ent_coef: 0.0
vf_coef: 0.0001
learning_rate: 0.0003
clip_range: 0.2
policy_kwargs:
  activation_fn: nn.ELU
  net_arch:
    pi:
    - 256
    - 128
    - 64
    vf:
    - 256
    - 128
    - 64
target_kl: 0.01
max_grad_norm: 1.0
```

One of the solutions that worked and I saw in an official NVIDIA tutorial was to change the physics of the 2 finger joints of the franka arm from *convexHull* to -> **convexDecomposition** (much better hitboxes). 

I parallelzized the training with my GPU, these were the used commands: 

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Lift-Cube-Franka-IK-Abs-v0 --num_envs 4096 --headless
```
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Lift-Cube-Franka-IK-Abs-v0 --num_envs 16 # to check training visually
```

This was the result: 
![[Franka levanta cubo.gif]]

However, the inference results were terrible. 