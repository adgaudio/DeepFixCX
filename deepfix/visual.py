import os
import torch as T
import numpy as np
def get_activation(name,activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_activations_vec_attn(model,device,x):
    
    from captum.attr import GradientShap
    from captum.attr import IntegratedGradients
    from captum.attr import NoiseTunnel
    from captum.attr import GuidedGradCam
    activation={}
    model.mlp.spatial_attn[1].register_forward_hook(get_activation('mlp.spatial_attn.1',activation))
    model.mlp.spatial_attn[0].register_forward_hook(get_activation('mlp.spatial_attn.0',activation))
    model.compression_mdl.wavelet_encoder.register_forward_hook(get_activation('compression_mdl.wavelet_encoder',activation))
    model.eval()


    ig=IntegratedGradients(model.mlp)
    nt=NoiseTunnel(ig)
    guided_gc=GuidedGradCam(model.mlp,model.mlp.spatial_attn[1])
    gradient_shap=GradientShap(model.mlp)
    x=x.to(device,non_blocking=True)
    op=model(x)
    grads=T.autograd.grad(op,model.mlp.spatial_attn[1].vec_attn,retain_graph=False)
    w=model.compression_mdl(x)
    w.requires_grad=True
    att_ig = ig.attribute(w)
    att_nt = nt.attribute(w, nt_type='smoothgrad',nt_samples=10)
    att_gc = guided_gc.attribute(w)
    att_gs = gradient_shap.attribute(w,T.randn(*w.size()).to(device))
    return x[0],activation['mlp.spatial_attn.0'][0],activation['mlp.spatial_attn.1'][0],activation['compression_mdl.wavelet_encoder'][0],grads[0],att_ig,att_nt,att_gc,att_gs

def visualize_activation(model:T.nn.Module,wi:T.nn.Module,device,img_num:int,x:T.tensor,y:int,y_hat:int,J:int,PS:int):
    from deepfix import plotting as P
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from matplotlib import colors
    
    os.makedirs('visualization_results',exist_ok=True)
    os.chdir('visualization_results')
    os.makedirs('Sample_'+str(img_num)+'/activations',exist_ok=True)
    os.makedirs('Sample_'+str(img_num)+'/saliency',exist_ok=True)

    img,act_before,act_after,wt,autograd_grad,ig,nt,gc,gs=get_activations_vec_attn(model,device,x)

    plt.figure()
    pt=P.plot_img_grid(wt[0][:25],cmap=cm.get_cmap('RdYlGn'),norm=colors.CenteredNorm())
    pt.suptitle('Wavelet packet transform')
    plt.savefig('./Sample_'+str(img_num)+'/wavelet_packet_transform_cn.jpg')
    plt.close()
    
    plt.figure()
    pt=P.plot_img_grid(wt[0][:25],cmap=cm.get_cmap('RdYlGn'))
    pt.suptitle('Wavelet packet transform')
    plt.savefig('./Sample_'+str(img_num)+'/wavelet_packet_transform.jpg')
    plt.close()

    plt.figure()
    plt.imshow(img[0].cpu().detach().numpy(),cmap=cm.get_cmap('RdYlGn'),norm=colors.CenteredNorm())
    plt.title('Sample image')
    
    plt.colorbar()
    plt.savefig('./Sample_'+str(img_num)+'/sample.jpg') 
    plt.close()
     
    layout_shape=(4**int(J),int(PS),int(PS)) #to get the wavelet packet layout shape
    rep = (320//(2**int(J)))//PS #the repeat factor for inverse wavelet
    act_b=act_before.reshape(*layout_shape).cpu().detach().numpy()
    act_a=act_after.reshape(*layout_shape).cpu().detach().numpy()
    
    inv_act=wi(T.tensor(np.expand_dims(np.expand_dims(np.repeat(act_a,rep,axis=1).repeat(rep,axis=2),0),0)).to(device)).cpu().detach().numpy()
    
    ig=ig.reshape(*layout_shape).cpu().detach().numpy()
    inv_ig=wi(T.tensor(np.expand_dims(np.expand_dims(np.repeat(ig,rep,axis=1).repeat(rep,axis=2),0),0),dtype=T.float).to(device)).cpu().detach().numpy() 
    ig_sum=np.mean(ig,0)


    autograd_grad=autograd_grad.reshape(*layout_shape).cpu().detach().numpy()
    print(autograd_grad.shape)
    inv_autograd=wi(T.tensor(np.expand_dims(np.expand_dims(np.repeat(autograd_grad,rep,axis=1).repeat(rep,axis=2),0),0),dtype=T.float).to(device)).cpu().detach().numpy()
    
    nt=nt.reshape(*layout_shape).cpu().detach().numpy()
    inv_nt=wi(T.tensor(np.expand_dims(np.expand_dims(np.repeat(nt,rep,axis=1).repeat(rep,axis=2),0),0),dtype=T.float).to(device)).cpu().detach().numpy()
    
    gc=gc.reshape(*layout_shape).cpu().detach().numpy()
    inv_gc=wi(T.tensor(np.expand_dims(np.expand_dims(np.repeat(gc,rep,axis=1).repeat(rep,axis=2),0),0),dtype=T.float).to(device)).cpu().detach().numpy()
    
    gs=gs.reshape(*layout_shape).cpu().detach().numpy()
    inv_gs=wi(T.tensor(np.expand_dims(np.expand_dims(np.repeat(gs,rep,axis=1).repeat(rep,axis=2),0),0),dtype=T.float).to(device)).cpu().detach().numpy()
    
    grad_sum=np.mean(autograd_grad,0)


    # Plotting Reconstructions of different saliency measures(inverse wavelet using gradients)

    plt.figure()
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    im=axs[0,0].imshow(img[0].cpu().detach().numpy(),cmap=cm.get_cmap('RdYlGn'))
    axs[0,0].set_title('Input image')
    fig.colorbar(im,ax=axs[0,0],shrink=0.25)


    im=axs[1,0].imshow(inv_ig[0][0],cmap=cm.get_cmap('RdYlGn'))
    axs[1,0].set_title('Integrated Grad')
    fig.colorbar(im,ax=axs[1,0],shrink=0.25)

    im=axs[0,1].imshow(inv_gc[0][0],cmap=cm.get_cmap('RdYlGn'))
    axs[0,1].set_title('Guided grad cam')
    fig.colorbar(im,ax=axs[0,1],shrink=0.25)
    
    im=axs[1,1].imshow(inv_nt[0][0],cmap=cm.get_cmap('RdYlGn'))
    axs[1,1].set_title('Noise Tunnel')
    fig.colorbar(im,ax=axs[1,1],shrink=0.25)

    fig.suptitle('Wavelet reconstructions')
    plt.savefig('./Sample_'+ str(img_num)+'/reconstruction_saliency.jpg')
    plt.close()
    

    #Plotting Reconstructions using activation and saliency

    plt.figure()
    fig,axs=plt.subplots(1,3,figsize=(12,8))
    im=axs[0].imshow(img[0].cpu().detach().numpy(),cmap=cm.get_cmap('RdYlGn'))
    axs[0].set_title('Input image')
    fig.colorbar(im,ax=axs[0],shrink=0.25)


    im=axs[1].imshow(inv_act[0][0],cmap=cm.get_cmap('RdYlGn'))
    axs[1].set_title('Using activation')
    fig.colorbar(im,ax=axs[1],shrink=0.25)

    im=axs[2].imshow(inv_autograd[0][0],cmap=cm.get_cmap('RdYlGn'))
    axs[2].set_title('Using Saliency')
    fig.colorbar(im,ax=axs[2],shrink=0.25)

    fig.suptitle('Inverse wavelet reconstructions')
    plt.savefig('./Sample_'+ str(img_num)+'/reconstruction.jpg')
    plt.close()

    
    #Autograd saliency (vec attn) 

    plt.figure()
    pl=P.plot_img_grid(autograd_grad[:25],cmap=cm.get_cmap('RdYlGn'))
    pl.suptitle('Saliency with respect Vec Attn')
    plt.savefig('./Sample_'+str(img_num)+'/saliency/saliency.jpg')
    plt.close()

    #Average saliency

    plt.figure()
    plt.imshow(grad_sum,cmap=cm.get_cmap('RdYlGn'))
    plt.title('Average Saliency with respect Vec Attn')
    plt.colorbar()
    plt.savefig('./Sample_'+str(img_num)+'/saliency/avg_saliency.jpg')
    plt.close()
    
    #IG saliency
    plt.figure()
    pl=P.plot_img_grid(ig[:25],cmap=cm.get_cmap('RdYlGn'))
    pl.suptitle('Saliency (IG) with respect Vec Attn')
    plt.savefig('./Sample_'+str(img_num)+'/saliency/saliency_ig.jpg')
    plt.close()

    #Average IG saliency
    plt.figure()
    plt.imshow(ig_sum,cmap=cm.get_cmap('RdYlGn'))
    plt.title('Average Saliency (IG) with respect Vec Attn')
    plt.colorbar()
    plt.savefig('./Sample_'+str(img_num)+'/saliency/avg_saliency_ig.jpg')
    plt.close()

    #Getting average activations
    act_b_sum=np.mean(act_b,0)
    act_a_sum=np.mean(act_a,0)

    #Before and after average activations    
    plt.figure()
    fig,axs=plt.subplots(1,3,figsize=(12,8))
    im=axs[0].imshow(img[0].cpu().detach().numpy(),cmap=cm.get_cmap('RdYlGn'))
    axs[0].set_title('Input image')
    fig.colorbar(im,ax=axs[0],shrink=0.25)
    
    '''axs[1,0].axis('off')
    labels='y = '+str(y)+' , y_hat = '+str(y_hat) 
    axs[1,0].text(0.5,0.5,labels)'''


    im=axs[1].imshow(act_b_sum,cmap=cm.get_cmap('RdYlGn'))
    axs[1].set_title('Before')
    fig.colorbar(im,ax=axs[1],shrink=0.25)

    im=axs[2].imshow(act_a_sum,cmap=cm.get_cmap('RdYlGn'))
    axs[2].set_title('After')
    fig.colorbar(im,ax=axs[2],shrink=0.25)

    fig.suptitle('Vector attention activations'+' y_hat='+str(y_hat)+ ' y='+str(y))
    plt.savefig('./Sample_'+ str(img_num)+'/activations/Attention_activations.jpg')
    plt.close()

    
    #Activations in wavelet packet layout    
    plt.figure()
    pl=P.plot_img_grid(act_a[:25],cmap=cm.get_cmap('RdYlGn'))
    pl.suptitle('Activation after vector attention')
    plt.savefig('./Sample_'+str(img_num)+'/activations/act_after.jpg')
    plt.close()


    plt.figure()
    pl=P.plot_img_grid(act_b[:25],cmap=cm.get_cmap('RdYlGn'))
    pl.suptitle('Activation before vector attention')
    plt.savefig('./Sample_'+str(img_num)+'/activations/act_before.jpg')
    plt.close()
    os.chdir("..")

