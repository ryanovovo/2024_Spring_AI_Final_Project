# 2024 Spring AI Final Project

## Requrements
### Hardware
Nvidia GPU with VRAM >= 8G

### Software
sudo apt install -y nvidia-cuda-toolkit  

conda create --name 2024_Spring_AI_Final_Project python=3.10

conda activate 2024_Spring_AI_Final_Project

conda install scikit-learn pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps trl peft accelerate bitsandbytes

## Run
python main.py

