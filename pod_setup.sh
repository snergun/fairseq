# Set up environmetn
conda create -n fairseq python=3.9 -y
source activate fairseq
pip install --upgrade "pip<24.1"
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install cython
pip install "omegaconf<2.1,>=2.0.5"
python setup.py build_ext --inplace
pip install --no-build-isolation -e .

# Downoad GBW model
cd examples/language_model
wget https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2
tar -xvf adaptive_lm_gbw_huge.tar.bz2
rm adaptive_lm_gbw_huge.tar.bz2
mv adaptive_lm_gbw_huge/model.pt adaptive_lm_gbw_huge/fq_model.pt

# Save fairseq checkpoint to only contain model weights
python3 -c "
import torch
cp = torch.load('adaptive_lm_gbw_huge/fq_model.pt')
torch.save({'model': cp['model']}, 'adaptive_lm_gbw_huge/model.pt')
print('Model conversion completed!')
"
# Save layer outputs and target words
cd ../..
MODEL_PATH=examples/language_model/adaptive_lm_gbw_huge
python -u fairseq_cli/eval_lm.py \
    $MODEL_PATH/data-bin \
    --path $MODEL_PATH/fq_model.pt \
    --sample-break-mode eos \
    --max-tokens 2048 \
    --save-layers -1 \
    --save-probs \
    --batch-size 64 \
    --results-path $MODEL_PATH/results \
    --gen-subset valid 