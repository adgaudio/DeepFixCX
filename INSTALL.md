
Code used to setup the environment:

    source /opt/anaconda/bin/activate
    conda create -n deepfixcx python=3.9
    conda activate deepfixcx
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

    conda install matplotlib numpy pandas scikit-learn pip tqdm pytables ipython seaborn
    conda install simple-parsing opencv -c conda-forge
    conda install pywavelets termcolor configargparse munch
    conda install -c conda-forge ray-tune  ray-default
    pip install --no-deps simplepytorch

    pip install adabound
    pip install pampy
    pip install efficientnet_pytorch
    pip install welford

    mkdir -p third_party
    git clone https://github.com/fbcotter/pytorch_wavelets third_party/pytorch_wavelets
    pushd third_party/pytorch_wavelets
    pip install -e . --no-deps
    popd

    # optional extras
    conda install pooch scikit-image

    # note: you might need to install redis for the ./bin/experiments.sh
    # conda install redis
