# LLM Finetuning: A Practical Guide
A comprehensive guide to understanding and implementing LLM finetuning, covering both theoretical foundations and practical implementation.

Refer to the [wiki](https://github.com/AmirYunus/finetune_LLM/wiki) for more details.

For code implementation, refer to [this Jupyter Notebook](lab.ipynb).

## Prerequisites and Virtual Environment
```bash
sudo apt update
sudo apt install cmake libcurl4-openssl-dev -y

conda create --prefix=venv python=3.11 pytorch-cuda=12.4 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
conda activate ./venv

python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m pip install --no-deps trl peft accelerate bitsandbytes
python -m pip install jupyter
```
