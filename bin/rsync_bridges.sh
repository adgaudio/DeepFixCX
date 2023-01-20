#!/bin/bash

# rsync -ave ssh ./data/BBBC038v1_microscopy bridgesdata:store/data/
rsync -ave ssh --exclude simplepytorch/simplepytorch.egg-info ../simplepytorch  bridgesdata:store/
# rsync -ave ssh ./bin ./dw ./dw2 ./svdsteering_centered ./svdsteering_noncentered ./dct2steering ./zero_weights bridgesdata:store/deep_wavelet
rsync -ave ssh --exclude __pycache__ ./bin ./waveletfix bridgesdata:store/waveletfix/
