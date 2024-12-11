import torch
import numpy as np
import scipy.io as sio
from torch.autograd import Variable
import os
from ourfunction import *
from Spa_downs import *
import time
from SSIM import *

from MSISRmodel import *
from thop import profile


os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
torch.cuda.empty_cache()


#######
datasetname ="Pavia"

method = "ourS"  ###"ourFusion","ourMSISR","ourS"
if datasetname != "realZY" and datasetname != "realTJC" and datasetname != "hyperions":
    factor = 8
else:
    factor = 3
ws = [7,2]
batch_size = 1
EPOCH = 30
Blind = True

if datasetname =="Pavia":
    Data = sio.loadmat('/home/longj/dataset/pavia/original_rosis'+'.mat')
    HSI = Data['HRHSI']
    P = sio.loadmat('/home/longj/dataset/SRF/L.mat')
    R = P['L']
    P = Variable(torch.unsqueeze(torch.from_numpy(P['L']),0)).type(torch.cuda.FloatTensor)
    patch_size = 10*factor
    patch_stride = 2*factor
    EPOCH = 30
elif datasetname =="CAVE":
    Data = sio.loadmat('/home/longj/dataset/CAVE/oil_painting_ms'+'.mat')  
    HSI = Data['Z_ori']#### numpy 512  512 31
    HSI = HSI.transpose(2,0,1)
    P = sio.loadmat('/home/longj/dataset/SRF/P_N_V2.mat')
    R = P['P']
    P = Variable(torch.unsqueeze(torch.from_numpy(P['P']),0)).type(torch.cuda.FloatTensor)
    patch_size = 20*factor
    patch_stride = 2*factor
    EPOCH = 20
elif datasetname =="realZY":
    Data = sio.loadmat('/home/longj/dataset/realdata/realZYnohr.mat')
    HSI  = Data['HSI'][0:256,0:256,:]  ### 512 512 76
    MSI =  Data['MSI'][0:256*3,0:256*3,:]   ### 1536 1536 8
    HSI = HSI.transpose(2,0,1)
    MSI = MSI.transpose(2,0,1)
    HSI =torch.from_numpy(HSI)
    MSI =torch.from_numpy(MSI)
    LR_HSI = HSI.unsqueeze(0).type(torch.cuda.FloatTensor)
    HR_MSI = MSI.unsqueeze(0).type(torch.cuda.FloatTensor)
    GT = HR_MSI
    patch_size = 20*factor
    patch_stride = 20*factor
    down_spa3 = Spa_Downs(
        HSI.shape[0], factor, kernel_type='gauss12', kernel_width=ws[0],
        sigma=ws[1],preserve_size=True
    ).type(torch.cuda.FloatTensor)
    EPOCH = 10 
elif datasetname =="realTJC":
    hsi= sio.loadmat('/home/longj/oldserver/data/real/realdata1.mat')  ### 333x422x224  999x1266x3
    LR_HSI = Variable(torch.unsqueeze(torch.from_numpy((hsi['HSI']).astype('float32')),0), requires_grad=False).type(torch.cuda.FloatTensor)
    HR_MSI = Variable(torch.unsqueeze(torch.from_numpy((hsi['MSI']).astype('float32')),0,), requires_grad=False).type(torch.cuda.FloatTensor)

    LR_HSI = LR_HSI[:, :, :, 5::5].permute(0,3,1,2)### 1x44x333x422
    HR_MSI = HR_MSI[:, :, :, :].permute(0,3,1,2)### 1x3x999x1266
    GT = HR_MSI
    patch_size = 40*factor
    patch_stride = 4*factor
    down_spa3 = Spa_Downs(
        LR_HSI.shape[1], factor, kernel_type='gauss12', kernel_width=ws[0],
        sigma=ws[1],preserve_size=True
    ).type(torch.cuda.FloatTensor)
    EPOCH = 10

elif datasetname =="hyperions":
    hsi= sio.loadmat('/home/longj/dataset/realdata/hyperions/HSIMSI20.mat')  ### 100x100x89 300x300x4
    LR_HSI = Variable(torch.unsqueeze(torch.from_numpy((hsi['HSI']).astype('float32')),0), requires_grad=False).type(torch.cuda.FloatTensor)
    HR_MSI = Variable(torch.unsqueeze(torch.from_numpy((hsi['MSI']).astype('float32')),0,), requires_grad=False).type(torch.cuda.FloatTensor)
    LR_HSI = LR_HSI.permute(0,3,1,2)### 1x89x100x100
    HR_MSI = HR_MSI.permute(0,3,1,2)### 1x4x300x300
    GT = HR_MSI
    patch_size = 20*factor
    patch_stride = 10*factor
    down_spa3 = Spa_Downs(
        LR_HSI.shape[1], factor, kernel_type='gauss12', kernel_width=ws[0],
        sigma=ws[1],preserve_size=True
    ).type(torch.cuda.FloatTensor)
    EPOCH = 15

else:
    print("Not have the dataset",datasetname)

if datasetname != "realZY" and datasetname != "realTJC" and datasetname != "hyperions":
    max_value = np.max(HSI)
    HSI = HSI / max_value
    HSI =torch.from_numpy(HSI)
    GT = HSI.unsqueeze(0).type(torch.cuda.FloatTensor)
    HR_MSI = torch.matmul(P,GT.reshape(-1,GT.shape[1],GT.shape[2]*GT.shape[3])).reshape(-1,P.shape[1],GT.shape[2],GT.shape[3])
    HR_MSI = Variable(HR_MSI, requires_grad=False).type(torch.cuda.FloatTensor)

    down_spa3 = Spa_Downs(
        HSI.shape[0], factor, kernel_type='gauss12', kernel_width=ws[0],
        sigma=ws[1],preserve_size=True
    ).type(torch.cuda.FloatTensor)

    LR_HSI = down_spa3(GT)
    LR_HSI = Variable(LR_HSI, requires_grad=False).type(torch.cuda.FloatTensor)



print("data load done!")

################Build Train dataset


if method == "ourS":
    halfsize = LR_HSI.shape[2]//2
    LR_HSITR  = LR_HSI[:,:,:,0:halfsize]
    HR_MSITR = HR_MSI[:,:,:,0:halfsize*factor]
    GTTR = GT[:,:,:,0:halfsize*factor]
elif method == "ourFusion":
    LR_HSITR  = LR_HSI
    HR_MSITR = HR_MSI
    GTTR = GT
else:
    halfsize = LR_HSI.shape[2]//2
    LR_HSITR  = GT[:,:,:,0:halfsize*factor]
    HR_MSITR = HR_MSI[:,:,:,0:halfsize*factor]
    GTTR = GT[:,:,:,0:halfsize*factor]

if Blind == True:
    if datasetname != "realZY" and datasetname != "realTJC" and datasetname != "hyperions":
        P = estR(LR_HSITR,HR_MSITR,factor)
    else: 
        P =estRforrealdata(LR_HSITR,HR_MSITR,factor)
 

dataset = PatchDataset(LR_HSITR, HR_MSITR,GTTR, patch_size, patch_stride, factor)
loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

############## Build model
lr  = 1e-3
model = MSISRNet(HR_MSI.shape[1], LR_HSI.shape[1],P).cuda()       
optimizer = torch.optim.Adam(model.parameters(),lr = lr)
n = 0
def LR_Decay(optimizer, n):
    lr_d = lr * (0.7**n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_d

print("Print space dimensions for HSI and MSI for Train:",LR_HSITR.shape,HR_MSITR.shape)

time_start = time.perf_counter()
for epoch in range(EPOCH):
    for i, (LR_HSI2, HR_MSI2, GT2, item) in enumerate(loader):
        out= model(HR_MSI2)
        if Blind == True:
            Yhsi,Ymsi = cpt_targetforblind(out,LR_HSI2,P,down_spa3)
            loss0 = build_lossblind(Yhsi, Ymsi, LR_HSI2, HR_MSI2,down_spa3) 
        else:
            Yhsi,Ymsi = cpt_target(out,down_spa3,P)
            loss0 = build_loss(Yhsi, Ymsi, LR_HSI2, HR_MSI2)
        
        optimizer.zero_grad()
        loss0.backward(retain_graph=True)
        optimizer.step()
    if epoch%5 == 0 :
        if datasetname != "realZY" and datasetname != "realTJC" and datasetname != "hyperions":
            psnr = PSNR_GPU(GT2.cpu(), out.detach().cpu())
            sam = SAM_GPU(GT2, out.detach())
            ssim = ssim_GPU(GT2, out.detach())
            print('At the {0}th epoch the Loss,PSNR,SAM, SSIM,are {1:.8f}, {2:.2f}, {3:.2f},{4:.2f}.'.format(epoch, loss0, psnr, sam,ssim))   
        else:
            print('At the {0}th epoch the Loss is {1:.8f}.'.format(epoch, loss0))
    if epoch%5 == 0:
        LR_Decay(optimizer, n) 
        n += 1
        print('Adjusting the learning rate by timing 0.8.')
        prev_loss0 = loss0
time_end = time.perf_counter()
train_time = time_end - time_start
time_start = time.perf_counter()
with torch.no_grad():
    out= model(HR_MSI)
time_end = time.perf_counter()
test_time = time_end - time_start
if datasetname != "realZY" and datasetname != "realTJC" and datasetname != "hyperions":
    psnr = PSNR_GPU(GT.cpu(), out.detach().cpu())
    sam = SAM_GPU(GT, out.detach())
    ssim = ssim_GPU(GT, out.detach())
    print('**************************The final PSNR,SAM, SSIM,are {0:.4f}, {1:.4f}, {2:.4f}.'.format( psnr, sam,ssim))
else:
    print('**************************The final Loss is {0:.8f}.'.format( loss0))


out = out.cpu().detach().numpy()
out = np.squeeze(out)
out = np.transpose(out, (1,2,0))
print("The size of the dimension of the printout:",out.shape)


HR_MSI2 = HR_MSI.unsqueeze(0)
flops, params = profile(model, HR_MSI2)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))


R2 = P.cpu().detach().numpy()
R2 = np.squeeze(R2)

print("The size of the dimension of the printout:",R2.shape)

if datasetname != "realZY" and datasetname != "realTJC" and datasetname != "hyperions":
    ckpt_model_filename = "./Results/"+ datasetname + "R_" + ".mat"
    sio.savemat(ckpt_model_filename,  {'R2': R2,'R': R })
    ckpt_model_filename = "./Results/"+ datasetname + "_" + ".mat"
    sio.savemat(ckpt_model_filename,  {'out': out,'traintime':train_time,'testtime':test_time })
else:
    ckpt_model_filename = "./Results/"+ datasetname + "R_" + ".mat"
    sio.savemat(ckpt_model_filename,  {'R2': R2})

    lrh = LR_HSITR.cpu().detach().numpy()
    lrh = np.squeeze(lrh)
    lrh = np.transpose(lrh, (1,2,0))
    hrm = HR_MSI.cpu().detach().numpy()
    hrm = np.squeeze(hrm)
    hrm = np.transpose(hrm, (1,2,0))
    print("The size of the dimension of the printout:",out.shape)
    ckpt_model_filename = "./Results/"+ datasetname + "_" + ".mat"
    sio.savemat(ckpt_model_filename,  {'out': out,'lrh': lrh,'hrm': hrm,'traintime':train_time,'testtime':test_time })
    