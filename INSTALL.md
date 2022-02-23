
Code used to setup the environment:

    source /opt/anaconda/bin/activate
    conda create -n deepfix python=3.9
    conda activate deepfix
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

    conda install matplotlib numpy pandas scikit-learn pip tqdm pytables ipython seaborn
    conda install simple-parsing -c conda-forge
    conda install pywavelets termcolor configargparse munch
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
