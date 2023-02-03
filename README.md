# DeepFixCX : Explainable Privacy-Preserving Image Compression for Medical Image Analysis

Code accompanying the paper.

[Paper on Wiley Journal of Data Mining and Knowledge Discovery](TODO)
[Paper arXiv](TODO)

![Graphical Abstract](/DeepFixCX_Graphical_Abstract.svg)
![Venn Diagram](/deepfix_venndiagram.jpg)
![Results](/deepfix_fig1_acc_vs_imr.jpg)

### Citation:

TODO: citation and bibtex.

### Reproducibility

All experiments are in `./bin/experiments.sh` and `./bin/experiments_extended.sh`

It's research quality code.  Note that the name was "DeepFix" and then I changed
it to DeepFixCX.  Please open a GitHub issue if there is a reproducibility
problem.

### Use the code

```
# compress and privatize images:
from deepfixcx.models import DeepFixCXImg2Img

# this may also be useful in projects:
from deepfix.models.wavelet_packet import WaveletPacket2d

# more control over the details:
from deepfixcx.models import DeepFixCXCompression, DeepFixCXReconstruct
```




    
