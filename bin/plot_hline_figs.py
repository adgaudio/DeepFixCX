import os
from matplotlib import pyplot as plt
import numpy as np
import torch as T
import torchvision.transforms as tvt
from deepfix.models.qthline import QTLineClassifier, RLine, HLine, QT


im = T.tensor(plt.imread('./data/CheXpert-v1.0-small/train/patient64533/study1/view1_frontal.jpg')/255, dtype=T.float)
im = tvt.CenterCrop((320,320))(im).reshape(1,1,320,320)

qrh = QTLineClassifier(
            RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))),
            QT(100, 9, split_mode='entropy'))
rh = QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))), None)
r = QTLineClassifier(RLine((320,320), nlines=200, seed=1), None)
h = QTLineClassifier(HLine(list(range(100,300,10)), 320), None)
heart = QTLineClassifier(RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=True, hlines=[]), None)



os.makedirs('tmpplot', exist_ok=True)
#
ax = plt.gca()
ax.axis('off')
ax.imshow(im.squeeze(), cmap='Greys_r')
plt.savefig('tmpplot/img_centercropped.png', dpi=300, bbox_inches='tight')

# hlines
# ... mask
ax = plt.gca()
ax.axis('off')
z = np.zeros_like(im.squeeze().numpy())
z[h.lines_fn.lines] = 1
ax.imshow(z, cmap='copper')
plt.savefig('tmpplot/mask_hline.png', dpi=300, bbox_inches='tight')
# ... im
ax.imshow(im.squeeze(), cmap='Greys_r')
z = np.zeros_like(im.squeeze().numpy())
z[h.lines_fn.lines] = im.squeeze()[h.lines_fn.lines]
ax.imshow(z**2, alpha=1, cmap='Greys_r')
plt.savefig('tmpplot/img_hline.png', dpi=300, bbox_inches='tight')

# rlines plot
# ... mask
ax = plt.gca()
ax.axis('off')
ax.imshow(r.lines_fn.arr.squeeze(), cmap='copper')
plt.savefig('tmpplot/mask_rline.png', dpi=300, bbox_inches='tight')
# ... img
ax.cla()
#  ax.imshow(im.squeeze(), cmap='Greys_r')
z = np.zeros_like(im.numpy()).squeeze()
z[r.lines_fn.arr.squeeze()] = im.squeeze()[r.lines_fn.arr.squeeze()]
ax.imshow(z.squeeze()**2, cmap='Greys_r')
plt.savefig('tmpplot/img_rline.png', dpi=300, bbox_inches='tight')

# heart plot
# ... mask
ax = plt.gca()
ax.axis('off')
ax.imshow(heart.lines_fn.arr.squeeze(), cmap='copper')
plt.savefig('tmpplot/mask_heart.png', dpi=300, bbox_inches='tight')
# ... img
ax = plt.gca() ; ax.cla() ; ax.axis('off')
#  ax.imshow(im.squeeze(), cmap='Greys_r')
z = np.zeros_like(im.numpy()).squeeze()
z[heart.lines_fn.arr.squeeze()] = im.squeeze()[heart.lines_fn.arr.squeeze()]
ax.imshow(z.squeeze()**2, cmap='Greys_r')
plt.savefig('tmpplot/img_heart.png', dpi=300, bbox_inches='tight')

# rhlines
# ... mask
ax = plt.gca()
ax.axis('off')
ax.imshow(rh.lines_fn.arr.squeeze(), cmap='copper')
plt.savefig('tmpplot/mask_rhline.png', dpi=300, bbox_inches='tight')
# ... img
ax = plt.gca() ; ax.cla() ; ax.axis('off')
#  ax.imshow(im.squeeze(), cmap='Greys_r')
z = np.zeros_like(im.numpy()).squeeze()
z[rh.lines_fn.arr.squeeze()] = im.squeeze()[rh.lines_fn.arr.squeeze()]
ax.imshow(z.squeeze()**2, cmap='Greys_r')
plt.savefig('tmpplot/img_rhline.png', dpi=300, bbox_inches='tight')

# qrhlines
# ... mask
ax = plt.gca()
ax.axis('off')
ax.imshow(qrh.lines_fn.arr.squeeze(), cmap='copper')
plt.savefig('tmpplot/mask_qrhline.png', dpi=300, bbox_inches='tight')
# ... img
ax = plt.gca() ; ax.cla() ; ax.axis('off')
#  ax.imshow(im.squeeze(), cmap='Greys_r')
z = np.zeros_like(im.numpy()).squeeze()
z[qrh.lines_fn.arr.squeeze()] = qrh.quadtree(im).squeeze()[qrh.lines_fn.arr.squeeze()]
ax.imshow(z.squeeze()**2, cmap='Greys_r')
plt.savefig('tmpplot/img_qrhline.png', dpi=300, bbox_inches='tight')

ax = plt.gca() ; plt.cla() ; ax.axis('off')
ax.imshow(qrh.quadtree(im).squeeze(), cmap='Greys_r')
plt.savefig('tmpplot/img_qtree.png', dpi=300, bbox_inches='tight')


# table on in-memory compression ratio
r.mlp = T.nn.Identity()
h.mlp = T.nn.Identity()
rh.mlp = T.nn.Identity()
qrh.mlp = T.nn.Identity()
heart.mlp = T.nn.Identity()
df = pd.Series({
    'HLine': h(im).numel() / im.numel(),
    'RLine': r(im).numel() / im.numel(),
    'Heart': heart(im).numel() / im.numel(),
    'RHLine': rh(im).numel() / im.numel(),
    'QRHLine': qrh(im).numel() / im.numel(),
})
print(df.to_latex())


