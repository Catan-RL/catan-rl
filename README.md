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
conda activate catan-rl
pip install -r requirements.txt
pip install -e .
```

On GCP instance:
```
# Install Git & dependencies for X11 forwarding over SSH
sudo apt install git libx11-6
# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init bash
```
