# Multi Agent Deep Reinforcement Learning for Simplified Settlers of Catan

To get started, first clone the repo:

```
git clone git@github.com:Catan-RL/catan-rl.git && cd catan-rl
# or
git clone https://github.com/Catan-RL/catan-rl.git && cd catan-rl
```

To install dependencies:

```
conda env create -f environment.yml
conda activate catan-rl-39

cd ..
git clone --branch v0.20.0-patched https://github.com/Catan-RL/gym.git && cd gym
pip install .

cd ..

git clone https://github.com/Replicable-MARL/MARLlib.git

cd catan-rl
pip install -r requirements.txt

cd ../MARLlib/marllib/patch
python add_patch.py -y
cd ../../../gym
pip install .

cd ../MARLlib/
cp ../catan-rl/requirements.txt ./
pip install .

cd ../catan-rl/
pip install -e .
```

To register the code with MARLlib:

```
./register_env.sh
cd ../MARLlib
```

On GCP instance:

```
# Install Git & dependencies for X11 forwarding over SSH
sudo apt install git libx11-6 libx11-dev xauth x11-xserver-utils mousepad xvfb # revise later
# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init bash
```

To enable SSH X11 forwarding, edit `/etc/ssh/sshd_config` and set `X11UseLocalhost no`, then `systemctl restart sshd`.

To connect to GCP instance:

```
gcloud compute ssh --ssh-flag="-Y" USERNAME@gpu-instance
su - shared # switch to shared user (password: shared)
```

GCP instance details:

```
NVIDIA T4 GPU
1 core 2 thread CPU
8 GB RAM
30 GB Disk Space
```


# DO NOT USE THESE INSTRUCTIONS, THEY ARE LEGACY
Install Marllib in a separate env:

First, initalize the env and install PettingZoo dependencies:

```
conda create -n marllib python=3.8
conda activate marllib
pip install pettingzoo==1.23.1
pip install supersuit==3.9.0
pip install pygame==2.3.0
```

Next, in a separate directory, clone the Marllib repo and run the following commands:

```
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install --upgrade pip
```

In Marllib's requirements.txt file, change the line:

```
gym==0.20.0
```

to

```
gym
```

then

```
pip install -r requirements.txt
```

Next, we need to add a patch and install marllib

```
pip install protobuf==3.20.3
cd MARLlib/marllib/patch
python add_patch.py -y
pip install marllib
pip install gym==0.22.0
```

You should be able to run marllib example code using

```
python mappo_example.py
```
