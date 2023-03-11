# DeepFixCX : Explainable Privacy-Preserving Image Compression for Medical Image Analysis

Code accompanying the paper.

[Open Access Paper on Wiley WIREs Journal of Data Mining and Knowledge Discovery](https://doi.org/10.1002/widm.1495)

<!-- [Paper arXiv](TODO) -->
<img alt="Graphical Abstract" src="/DeepFixCX_Graphical_Abstract.png" width="100%" />
<p float="left">
<img src="/deepfix_venndiagram.jpg" width="53%" />
<img src="/deepfix_fig1_acc_vs_imr.jpg" width="46%" />
</p>

### Citation:

> Gaudio, A., Smailagic, A., Faloutsos, C., Mohan, S., Johnson, E., Liu, Y., Costa, P., & Campilho, A. (2023). DeepFixCX: Explainable privacy-preserving image compression for medical image analysis. WIREs Data Mining and Knowledge Discovery, e1495. https://doi.org/10.1002/widm.1495

        @article{deepfixcx,
            author = {Gaudio, Alex and Smailagic, Asim and Faloutsos, Christos and Mohan, Shreshta and Johnson, Elvin and Liu, Yuhao and Costa, Pedro and Campilho, Aurélio},
            title = {DeepFixCX: Explainable privacy-preserving image compression for medical image analysis},
            journal = {WIREs Data Mining and Knowledge Discovery},
            volume = {n/a},
            number = {n/a},
            pages = {e1495},
            keywords = {compression, deep networks, explainability, medical image analysis, privacy, wavelets},
            doi = {https://doi.org/10.1002/widm.1495},
            url = {https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.1495},
            eprint = {https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/widm.1495},
        }


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
