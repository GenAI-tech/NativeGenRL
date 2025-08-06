conda create -n ngrl -y python=3.11
conda activate ngrl
pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements.txt
