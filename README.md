# DeepFixCX : Explainable Privacy-Preserving Image Compression for Medical Image Analysis

Code accompanying the paper.

[Paper on Wiley Journal of Data Mining and Knowledge Discovery](TODO)
[Paper arXiv](TODO)

<img alt="Graphical Abstract" src="/DeepFixCX_Graphical_Abstract.svg" width="100%" />
<img src="/deepfix_venndiagram.jpg" width="45%" /> <img src="/deepfix_fig1_acc_vs_imr.jpg" width="45%" />

### Citation:

TODO: citation and bibtex.

### Reproducibility

All experiments are in `./bin/experiments.sh` and `./bin/experiments_extended.sh`

It's research quality code.  Please open a GitHub issue if there is a reproducibility
problem.  Note that the name was "DeepFix" and then we changed it to "DeepFixCX".


### Installation

```
# cd into the root of the repository.

# Install libraries
pip install -r requirements.txt
pip install .  # install deepfix if you want.
```

### Use the API

Comments are in the docstrings.
```
# DeepFixCX:  Compress and privatize images:
from deepfixcx.models import DeepFixCXImg2Img

# Wavelet Packet layer may be useful in projects:
from deepfixcx.models.wavelet_packet import WaveletPacket2d

# DeepFixCX: just the compression or reconstruction parts.
from deepfixcx.models import DeepFixCXCompression, DeepFixCXReconstruct
```
