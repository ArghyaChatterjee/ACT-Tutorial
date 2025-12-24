# ACT Tutorial
A Tutorial on Action Chunking with Transformer. This tutorial uses code and data from original work [Open-TeleVision](https://robot-tv.github.io/).

## Requirements
This repository has been tested with ubuntu 24.04, Python 3.8, cuda 12.8 and nvidia RTX 4070.

## Clone the repo
```
git clone https://github.com/ArghyaChatterjee/ACT-Tutorial.git
```

## Installation
Activate conda environment and install packages:
```bash
cd ACT-Tutorial
source ~/miniconda3/bin/activate
conda create -n tv python=3.8
conda activate tv
conda install -c conda-forge av ffmpeg
# pip3 install -r requirements.txt
pip3 install -r requirements.txt --no-deps
cd act/detr 
pip3 install -e .
```

Install ZED sdk: https://www.stereolabs.com/developers/release/

Install ZED Python API:
```bash
cd /usr/local/zed/ 
python3 get_python_api.py
```

Install Isaac Gym: 

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym). Extract the zip file and copy the folder `isaacgym` inside `ACT-Tutorial`. Go inside `ACT-Tutorial`:
```bash
cd ~/ACT-Tutorial/isaacgym/python/
pip3 install -e .
```

Clone Dinov2 for Python 3.8:
```bash
cd ~/ACT-Tutorial
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
git checkout b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8
```

## Training Guide
1. Download dataset from [here](https://drive.google.com/drive/folders/11WO96mUMjmxRo9Hpvm4ADz7THuuGNEMY?usp=sharing). There are in total 5 datasets namely `00-can-sorting`, `01-can-insertion`, `02-folding`, `03-unloading`, and `04-gr1-can-sorting`.

2. Place the downloaded dataset in ``data/recordings/``. Each dataset contains observation states of multiple zed `svo` (left and right image pair) and corresponding `hdf5` (action sequences/joint positions) as episodes. 

3. Process the specified dataset for training using ``scripts/post_process.py``.
```
python post_process.py
```
This will create a folder inside every dataset named `processed` where corresponding `svo` and `hdf5` files for each episodes will be merged to produce single `hdf5` files. 

4. You can verify the observation states (image and action sequences) of a specific episode in the dataset using ``scripts/replay_demo.py``.
```
python replay_demo.py
```

<div align="center">
    <a href="https://www.youtube.com/watch?v=PThx8t9bFxQ">
        <img src="img/replay.gif" width="700">
    </a>  
</div>

5. To train ACT, run:
```
python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt
```

6. After training, save jit for the desired checkpoint:
```
python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt\
                               --save_jit --resume_ckpt 25000
```

7. You can visualize the action states with trained policy with inputs from dataset using ``scripts/deploy_sim.py``, example usage:
```
python deploy_sim.py --taskid 00 --exptid 01 --resume_ckpt 25000
```

## Description:

After expansion, full h1 robot qpos = 51 DOF 

### Observation State (inputs to the policy)
A) Joint Positions `qpos` → **observation state**

From `imitate_episodes.py` and `utils.load_data()`:

```python
state_dim = 26
```

This is:

```text
qpos_t ∈ ℝ²⁶
```

✔ Comes from:

```python
data['observation.state']
```

✔ Semantics:

* Robot joint positions at time **t**
* Low-dimensional, compact representation
* **Does NOT include every physical joint**
* Typically:

  * left arm
  * left hand 
  * right arm
  * right hand
  * may be torso / base

B) Images → **observation state**

Also part of the observation:

```python
image_data.shape = (B, 2, 3, H, W)
```

Processed by:

* **DINOv2 backbone**
* Produces visual tokens

Role:

* Images tell the policy **what the world looks like**



So, Observation = `{ qpos_t , left_image_t , right_image_t }`


### Action state (outputs of the policy)

A) `action_dim = 28` → **action state**

From config:

```python
action_dim = 28
```

This is:

```text
action_t ∈ ℝ²⁸
```

✔ Comes from:

```python
data['qpos_action']
```

✔ Predicted by ACT:

```python
pred_actions.shape = (chunk_size, 28)
```

✔ Represents:

> **Desired joint targets to command the robot next**


### Observation (`qpos`, 26D)

* What the robot **is**
* Reduced state
* Does **not** include detailed finger joints

### Action (`action`, 28D)

* What the robot **should do**
* Includes:

  * left arm (7)
  * right arm (7)
  * **compressed left hand (6 → 12 joints)**
  * **compressed right hand (6 → 12 joints)**

### Action vector breakdown

From `convert_h1_qpos()`:

| Action indices  | Meaning                                           |
| --------------- | ------------------------------------------------- |
| `action[0:7]`   | Left arm (7 DOF)                                  |
| `action[7:13]`  | Left hand **compressed** (6 scalars → 12 joints)  |
| `action[13:20]` | Right arm (7 DOF)                                 |
| `action[20:26]` | Right hand **compressed** (6 scalars → 12 joints) |
| `action[26:28]` | (unused / padding / legacy)                       |

That’s why:

```python
action_dim = 28
```


At time **t**:

1. Observation:

   ```
   obs_t = {qpos_t (26D), left_img_t, right_img_t}
   ```

2. Policy prediction (chunk size = 60):

   ```
   actions_t:t+59 = ACT(obs_t)  # shape (60, 28)
   ```

3. Execution:

   ```
   apply action_t = actions[0]
   ```



