import matplotlib.pyplot as plt
import scipy.stats
import torch as T
import torch.nn as nn


#  class QuadTree(T.nn.Module):
#      def __init__(self):
#          super().__init__()

#      def forward(self, x:T.Tensor):
#          """
#          Args:
#              x: minibatch of images, of shape (B,C,H,W)
#          """

#      def decide_if_split(self, x):
#          x.std()

#      def split(self, x):
#          #  split into quadrants
#          H,W = x.shape[-2:]

#          a = np.slice[:H//2]
#          x[:H//2, 
#          W//2
#          return x

def get_kde(img):
    assert img.max() <= 1
    bins = T.histc(img.reshape(-1), bins=256, min=0, max=1)
    kde = T.nn.functional.conv1d(bins.unsqueeze(0).unsqueeze(0), img.new_tensor([[[1., 1., 1.]]]), padding=1).squeeze()
    assert bins.shape == kde.shape
    kde = kde/kde.sum()
    return kde


class QuadTree:
    def insert(self, img, level, thresh, device:str='cpu', split_mode='entropy'):
        assert split_mode in {'std', 'entropy'}
        self.split_mode = split_mode
        #  self.kde = get_kde(img) if (kde is None and split_mode == 'entropy') else kde
        self.device = device
        self.level = level
        self.mean = T.mean(img)
        self.area=T.prod(T.tensor(img.shape))
        self.resolution = (img.shape[0], img.shape[1])
        self.leaf_node = True

        #  if img.shape[0]>=2 and img.shape[1]>=2 and self.weighted_average(img)>=thresh and img.shape[0]==img.shape[1]:
            #  split_img = self.split_into_4(img)
        if (img.shape[0]>1 or img.shape[1]>1) and self.weighted_average(img)>=thresh:
            split_img = self.split_tensor(img)
            self.leaf_node = False
            self.north_west = QuadTree().insert(split_img[0], level=level+1, thresh=thresh) #, kde=self.kde)
            self.north_east = QuadTree().insert(split_img[1], level=level+1, thresh=thresh) #, kde=self.kde)
            self.south_west = QuadTree().insert(split_img[2], level=level+1, thresh=thresh) #, kde=self.kde)
            self.south_east = QuadTree().insert(split_img[3], level=level+1, thresh=thresh) #, kde=self.kde)
        return self

    def get_image(self, level, map_type='mean'):
        assert map_type in {'mean', 'level'}
        if(self.leaf_node or self.level == level):
            return T.tile(T.tensor(self.level if map_type == 'level' else self.mean), (1,1,self.resolution[0], self.resolution[1]))
        return self.concatenate4(
            self.north_west.get_image(level, map_type=map_type), 
            self.north_east.get_image(level, map_type=map_type),
            self.south_west.get_image(level, map_type=map_type),
            self.south_east.get_image(level, map_type=map_type))

    def split_into_4(self,image):
        if(image.shape[0]%2==0):
            kernel_size=image.shape[0]//2
            patches = image.unfold(0, kernel_size,kernel_size).unfold(1, kernel_size,kernel_size)
            patches = patches.contiguous().view(-1,kernel_size, kernel_size)
            return patches
        else:
            half_size=(image.shape[0]//2)+1
            split1=T.split(image,half_size)
            split2=T.split(split1[0],half_size,1)
            split3=T.split(split1[1],half_size,1)
            return (split2[0],split2[1],split3[0],split3[1])

    def split_tensor(self, img):
        k=T.tensor_split(img, 2, dim=0)
        out=[]
        for i in k:
            l=T.tensor_split(i,2,dim=1)
            out.append(l[0])
            out.append(l[1])
        return out

    def weighted_average(self,img):
        #  hist=T.histc(T.flatten(img),bins=256,min=0,max=1)
        #  total = T.sum(hist)
        #  error = value = 0
        #  vector=(T.arange(256, device=self.device).float()/255)
        #  if total > 0:
        #      value = T.mean(img)
        #      error= T.sum(T.mul(hist,T.square(vector-value)))/total
        #      error = error ** 0.5
        #  assert T.allclose(error, img.std(unbiased=True), 1e-4, 1e-4) or T.allclose(error, img.std(unbiased=False), 1e-4, 1e-4)
        if self.split_mode == 'std':
            error = img.std()  #much faster, and the same.
        elif self.split_mode == 'entropy':
            # using entropy of this img in context of whole image
            #  x = self.kde[T.bucketize(img.reshape(-1), T.linspace(0,1,256))]
            #  _error = -1. * (x * T.log2(x)).sum()
            # using entropy of this img patch
            kde = get_kde(img)
            x = kde[T.bucketize(img.reshape(-1), T.linspace(0,1,256, device=img.device))]
            error = -1. * (x * T.log2(x)).sum()
            # combine entropies.  large value means more "boring" and "uniform"
            error = error#min(_error, error)
        else:
            raise NotImplementedError(self.split_mode)
        #  print(error)
        return error

    def concatenate4(self, north_west, north_east, south_west, south_east):
        top = T.cat((north_west, north_east), axis=3)
        bottom = T.cat((south_west, south_east), axis=3)
        return T.cat((top, bottom), axis=2)

    def __call__(self, matrix):
        tree = self.insert(matrix, level=0, thresh=self.thresh)
        return tree.get_image(level)


class QT(nn.Module):
    def __init__(self,thresh, split_mode='std'):
        super(QT,self).__init__()
        self.thresh=thresh
        self.quad=QuadTree()
        self.split_mode=split_mode

    def forward(self,x,level):
        out=[]
        for i in range(x.shape[0]):
            tree=self.quad.insert(x[i,0,:,:], level=0, thresh=self.thresh, split_mode=self.split_mode)
            out.append(tree.get_image(level, map_type='mean').cpu())
            fig, (a,b,c) = plt.subplots(1,3, clear=True, num=1)
            img = x[i,0].cpu()
            a.imshow(img, cmap='Greys')
            b.imshow(out[-1].squeeze(), cmap='Greys')
            #  c.imshow(tree.get_image(level, map_type='level').squeeze(), cmap='Greys')
            from skimage.metrics import structural_similarity as ssim
            #  c.imshow(ssim(x[i,0].cpu().numpy(),
                 #  out[-1].squeeze().cpu().numpy(), full=True)[1])
            c.imshow(img - out[-1].squeeze(), cmap='Greys')
            plt.show(block=False)
            plt.pause(1.001)
        return T.cat(tuple(out), dim=0)

if __name__ == "__main__":
    import glob
    from deepfix.train import get_dset_chexpert
    from matplotlib import pyplot as plt
    plt.ion()
    #  plt.gca()

    dct, _ = get_dset_chexpert(small=True, labels='Cardiomegaly', epoch_size=15000)
    qt = QT(5, split_mode='entropy').cuda()
    #  qt = QT(.2, split_mode='std')

    for mb in dct['train_loader']:
        z = qt(mb[0].cuda(), level=9)
        #  for im, qim in zip(mb[0], z):
            #  fig, (a,b) = plt.subplots(1,2, clear=True, num=1)
            #  a.imshow(im.squeeze(), cmap='Greys')
            #  b.imshow(qim.squeeze(), cmap='Greys')
            #  plt.show(block=False)
            #  plt.pause(3)
        break

    #  for fp in glob.glob('./data/CheXpert-v1.0-small/valid/*/*/*jpg'):
        #  im = plt.imread(fp)
        #  print(im.max(), im.shape)

