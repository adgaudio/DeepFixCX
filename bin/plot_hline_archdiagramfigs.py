import os
from matplotlib import pyplot as plt
import lzma
import numpy as np
import seaborn as sns
import subprocess
import pandas as pd
import torch as T
import torchvision.transforms as tvt
from deepfix.models.qthline import QTLineClassifier, RLine, HLine, QT
from deepfix.models.median_pooling import MedianPool2d
import cv2
import gzip

plt.rcParams.update({"text.usetex": True,})


im = T.tensor(plt.imread('./data/CheXpert-v1.0-small/train/patient64533/study1/view1_frontal.jpg')/255, dtype=T.float)
im = tvt.CenterCrop((320,320))(im).reshape(1,1,320,320)
#
qrh = QTLineClassifier(
            RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))),
            QT(100, 9, split_mode='entropy'))
rh = QTLineClassifier(RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))), None)
r = QTLineClassifier(RLine((320,320), nlines=200, seed=1), None)
h = QTLineClassifier(RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=False, hlines=list(range(100,300,10))), None)
heart = QTLineClassifier(RLine((320,320), nlines=0, zero_top_frac=0, seed=1, heart_roi=True, hlines=[]), None)
mrh = QTLineClassifier(
        RLine((320//2,320//2), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100//2,300//2,10//2))),
        quadtree=MedianPool2d(kernel_size=12, stride=2, same=True))
# if we used stride=1, the median pooling gives only minimal odr improvement and same imr as `rh`.
#  mrh = QTLineClassifier(
#          RLine((320,320), nlines=200, zero_top_frac=0, seed=1, heart_roi=True, hlines=list(range(100,300,10))),
#          quadtree=MedianPool2d(kernel_size=12, stride=1, same=True))


# Detour:  On-disk and in-memory compression ratio and fig1 plot for it.
# ... write the original center cropped image (jpg 95% is best and smaller than original)
assert cv2.imwrite('tmp.jpg', (im.squeeze()*255).round().numpy().astype('uint8'), [cv2.IMWRITE_JPEG_QUALITY, 95])
#  with lzma.open('tmp.xz', 'wb') as lzf:
#      np.save(file=lzf, arr=(im.squeeze()*255).round().numpy().astype('uint8'))
#  f = gzip.GzipFile("tmp.npy.gz", "w")
#  np.save(file=f, arr=im)
#  f.close()
# ... write the HeartSpot compressed img.
odr_imr = []
for i,qtlclassifier in enumerate((h,r,heart,rh,mrh)):
    print(i)
    if qtlclassifier.quadtree:
        z = qtlclassifier.quadtree(im)
    else:
        z = im
    z = qtlclassifier.lines_fn(z)
    # save as flat gzip vector
    #  f = gzip.GzipFile("tmp1.npy.gz", "w")
    #  np.save(file=f, arr=z)
    #  f.close()
    # lzma
    with lzma.open('tmp1.xz', 'wb') as lzf:
        np.save(file=lzf, arr=z)
        #  lzf.write(z.
    # numpy DEFLATE
    #  np.savez_compressed('tmp1.npz', z)
    # save as jpg
    z2 = z.new_zeros(qtlclassifier.lines_fn.arr.shape)
    z2[qtlclassifier.lines_fn.arr] = z
    cv2.imwrite('tmp1.jpg', (z2.squeeze()*255).round().numpy().astype('uint8'), [cv2.IMWRITE_JPEG_QUALITY, 95])
    #
    #  os.system('du -sb ./data/CheXpert-v1.0-small/train/patient64533/study1/view1_frontal.jpg tmp.xz tmp.jpg tmp.npy.gz tmp1.npy.gz tmp1.jpg tmp1.xz')
    #
    # REPORT ODR numbers when saving as gzip or as jpeg by uncommenting line below
    #
    a,b = subprocess.check_output('du -sb tmp.jpg tmp1.xz', shell=True).decode().strip().split('\n')
    #  a,b = subprocess.check_output('du -sb tmp.jpg tmp1.jpg', shell=True).decode().strip().split('\n')
    odr_imr.append({
        'Model': i,
        'On-Disk Compression Ratio': int(b.split('\t')[0])/int(a.split('\t')[0]),
        'In-Memory Compression Ratio': z.numel() / im.numel()
    })
    print(odr_imr[-1])
df = pd.DataFrame(odr_imr)
df['Model'].replace({0: 'HLine (Ours)', 1: 'RLine (Ours)', 2: 'Heart (Ours)', 3: '(RH)Line+Heart (Ours)',
                     4: 'Median+(RH)Line+Heart (Ours)'}, inplace=True)
fig, ax = plt.subplots(figsize=(4,2.5), dpi=200)
ax.scatter(1, 1, c='Gray', s=40*2, label='Baseline DenseNet121')
sns.scatterplot(x='On-Disk Compression Ratio', y='In-Memory Compression Ratio', hue='Model', data=df, s=24*4, ax=ax, legend=None, palette='tab10')
#  ax.hlines(1, 0.1, 1, colors='gray', linewidth=1, label='Baseline DenseNet121')
#  ax.vlines(1, 0, 1, colors='gray', linewidth=1, label=None)
mdl = 'Median+(RH)Line+Heart (Ours)'
xytext=(
    df.set_index('Model').loc[mdl, 'On-Disk Compression Ratio'],
    df.set_index('Model').loc[mdl, 'In-Memory Compression Ratio'],
)
ax.scatter(*xytext, color=plt.cm.tab10(4), s=40*2, label=r'\textit{HeartSpot} '+mdl.replace(' (Ours)', ''))
ax.legend(loc='lower right', ncol=1)
ax.annotate("", xytext=(1, 1), xy=xytext, arrowprops=dict(arrowstyle='->', lw=1), fontsize=26)
ax.text(.05, .70, '$14x$ Fewer Pixels\n$5x$ Smaller Filesize',
        bbox={'alpha': .7, 'color': 'white'},
        horizontalalignment='left', verticalalignment='center',
        transform=ax.transAxes, fontsize=26)
fig.savefig('hline_imr_vs_odr.png', bbox_inches='tight')
df.set_index('Model').to_csv('hline_odr_imr.csv')
print(df.set_index('Model').round(2)*100)
print((1/df.set_index('Model')).round(2))

import sys ; sys.exit()



### pictures for main architecture diagram figure.

fig, ax = plt.subplots(dpi=300, figsize=(4,4), clear=True)

os.makedirs('tmpplot', exist_ok=True)
#
ax = plt.gca()
ax.axis('off')
ax.imshow(im.squeeze(), cmap='Greys_r')
plt.savefig('tmpplot/img_centercropped.png', dpi=300, bbox_inches='tight')

# median
ax = plt.gca()
ax.axis('off')
ax.imshow(MedianPool2d(kernel_size=12, stride=2, same=True)(im).squeeze(), cmap='Greys_r')
plt.savefig('tmpplot/img_medianpool.png', dpi=300, bbox_inches='tight')

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
ax.cla() ; ax.axis('off')
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


