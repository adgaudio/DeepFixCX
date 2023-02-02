from deepfixcx.models.waveletmlp import DeepFixCXImg2Img
from matplotlib import pyplot as plt


dfx = DeepFixCXImg2Img(1, 2, 19, wavelet='db1', patch_features='l1', restore_orig_size=True)
xray = plt.imread('data/CheXpert-v1.0-small/valid/patient64542/study1/view1_frontal.jpg')/255
xray2 = dfx(T.tensor(xray, dtype=T.float).unsqueeze_(0).unsqueeze_(0)).numpy().squeeze()
plt.imshow(xray2, cmap='gray')
plt.imsave('results/plots/chexpert_patient64542_J2P19_recon.png', xray2, cmap='gray')


dfc = DeepFixCXImg2Img(3, J=7, P=22, wavelet='db1', patch_features='l1', restore_orig_size=True)
cervix = plt.imread('data/intel_mobileodt_cervical/test/test/6.jpg').transpose(2,0,1)/255
cervix2 = dfc(T.tensor(cervix, dtype=T.float).unsqueeze_(0)).numpy().squeeze().transpose(1,2,0)
plt.imshow(cervix2.clip(0,1), cmap='gray')
plt.imsave('results/plots/intelmobileodt_patient6_J3P7_recon.png', cervix2.clip(0,1), cmap='gray')

dfg = DeepFixCXImg2Img(3, J=1, P=59, wavelet='db1', patch_features='l1', restore_orig_size=True)
glau = plt.imread('data/kim_eye/advanced_glaucoma/1.png').transpose(2,0,1)
glau2 = dfg(T.tensor(glau, dtype=T.float).unsqueeze_(0)).numpy().squeeze().transpose(1,2,0)
plt.imshow(glau2.clip(0,1), cmap='gray')
plt.imsave('results/plots/kimeye_adv1_recon.png', glau2.clip(0,1), cmap='gray')
