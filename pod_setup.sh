conda create -n fairseq python=3.9 -y
source activate fairseq
pip install --upgrade "pip<24.1"
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118     --index-url https://download.pytorch.org/whl/cu118
pip install cython
pip install "omegaconf<2.1,>=2.0.5"
python setup.py build_ext --inplace
pip install --no-build-isolation -e .