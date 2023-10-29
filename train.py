import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from generators import *
from vtunet.build_vt_unet import build_vt_unet
from discriminators import *
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from init import Options
from utils.utils import *
from logger import *
import pandas as pd
import numpy as np
from thop import profile
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity
from models.generators import *
from models.discriminators import *


# -----  Loading the init options -----
print(torch.cuda.is_available())
print(torch.__version__)  #
# aa

opt = Options().parse()
min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))

if opt.gpu_ids != '-1':
    num_gpus = len(opt.gpu_ids.split(','))
else:
    num_gpus = 0
print('number of GPU:', num_gpus)
# -------------------------------------



# -----  Loading the list of data -----
train_list = create_list(opt.data_path)
val_list = create_list(opt.val_path)
# SECT_list = create_SECT_list(opt.val_path)

for i in range(opt.increase_factor_data):  # augment the data list for training

    train_list.extend(train_list)
#     val_list.extend(val_list)

print('Number of training patches per epoch:', len(train_list))
print('Number of validation patches per epoch:', len(val_list))
# -------------------------------------




# -----  Transformation and Augmentation process for the data  -----
trainTransforms = [
            NiftiDataset.Resample(opt.new_resolution, opt.resample),
            NiftiDataset.Augmentation(),
            NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
            NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
            ]

train_set = NifitDataSet(train_list, direction=opt.direction, transforms=trainTransforms, train=True)    # define the dataset and loader
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)  # Here are then fed to the network with a defined batch size
# -------------------------------------





# -----  Creating the Generator and discriminator -----
generator = build_netG(opt)
# generator =  build_vt_unet().cuda()
discriminator = build_netD(opt)
check_dir(opt.checkpoints_dir)

# -----  Pretrain the Generator and discriminator -----
if opt.resume:
    generator.load_state_dict(new_state_dict(opt.generatorWeights))
    discriminator.load_state_dict(new_state_dict(opt.discriminatorWeights))
    print('Generator and discriminator Weights are loaded')
else:
    pretrainW = './checkpoints/g_pre-train.pth'
    if os.path.exists(pretrainW):
        generator.load_state_dict(new_state_dict(pretrainW))
        print('Pre-Trained G Weight is loaded')
# -------------------------------------





criterionMSE = nn.MSELoss()  # nn.MSELoss()
criterionGAN = GANLoss()
criterion_pixelwise = nn.L1Loss()
# -----  Use Single GPU or Multiple GPUs -----
if (opt.gpu_ids != -1) & torch.cuda.is_available():
    use_gpu = True
    generator.cuda()
    discriminator.cuda()
    criterionGAN.cuda()
    criterion_pixelwise.cuda()
    criterionMSE.cuda()

    if num_gpus > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

optim_generator = optim.Adam(generator.parameters(), betas=(0.5,0.999), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), betas=(0.5,0.999), lr=opt.discriminatorLR)
net_g_scheduler = get_scheduler(optim_generator, opt)
net_d_scheduler = get_scheduler(optim_discriminator, opt)
# -------------------------------------



# -----  Training Cycle -----
print('Start training :) ')


log_name = opt.task + '_' + opt.netG + '_' + opt.netD
print("log_name: ", log_name)
f = open(os.path.join('result/2D/unet_gan/' + opt.task + '/', log_name + ".txt"), "w")


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for batch_idx, (data, label, filename) in enumerate(train_loader):
  

        real_a = data
        real_b = label
        real_a = real_a.cuda()

        # real_a = real_a[0,0].cpu().detach().numpy()
        # print(real_a.shape)
        # print(np.sum(real_a))

        

        # flops = FlopCountAnalysis(generator, real_a.cuda())
        # print("FLOPs: ", flops.total())
        # # total = sum([param.nelement() for param in generator.parameters()])
        # # print("params: ", summary(generator, [128,128,128]))
        # print("params: ", stat(generator, [128,128,128]))
        # # print("params: ", total)
        # aa

        if use_gpu:                              # forward
            real_b = real_b.cuda()
            fake_b = generator(real_a.cuda())    # generate fake data
            # fake_b = fake_b[0,0].cpu().detach().numpy()
            # print(fake_b.shape)
            # print(np.sum(fake_b))
            # real_a = real_a.cuda()
        else:
            fake_b = generator(real_a)



        

        

        ######################
        # (1) Update D network
        ######################
        optim_discriminator.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = discriminator.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = discriminator.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # Combined D loss
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        mean_discriminator_loss += discriminator_loss
        discriminator_loss.backward()
        optim_discriminator.step()

        ######################
        # (2) Update G network
        ######################

        optim_generator.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = discriminator.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterion_pixelwise(fake_b, real_b) * opt.lamb
        # loss_g_l1 = criterionMSE(fake_b, real_b) * opt.lamb

        generator_total_loss = loss_g_gan + loss_g_l1
        # generator_total_loss = loss_g_l1

        mean_generator_total_loss += generator_total_loss
        generator_total_loss.backward()
        optim_generator.step()


        ######### Status and display #########
        sys.stdout.write(
            '\r [%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss: %.4f' % (
                epoch, (opt.niter + opt.niter_decay + 1), batch_idx, len(train_loader),
                discriminator_loss, generator_total_loss))
        # print('\r [%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss: %.4f' % (
        #         epoch, (opt.niter + opt.niter_decay + 1), batch_idx, len(train_loader),
        #         discriminator_loss, generator_total_loss), file=f)

    update_learning_rate(net_g_scheduler, optim_generator)
    update_learning_rate(net_d_scheduler, optim_discriminator)
 


    if epoch % opt.save_fre == 0:
        save_path = os.path.join('result/2D/unet_gan/' + opt.task + '/', 'chk_'+str(epoch))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        rec = list()
        total_mse, total_nrmse ,total_psnr, total_ssim = [],[],[],[]
        original_total_mse, original_total_nrmse ,original_total_psnr, original_total_ssim = [],[],[],[]

        ##### Logger ######

        valTransforms = [
            # NiftiDataset.Resample(opt.new_resolution, opt.resample),
            # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
            # NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
        ]

        val_set = NifitDataSet(val_list, direction=opt.direction, transforms=valTransforms, test=True)
        val_loader = DataLoader(val_set, batch_size= 1, shuffle=False, num_workers=opt.workers)

        # SECT_set = NifitDataSet(SECT_list, direction=opt.direction, transforms=valTransforms, test=True)
        # SECT_loader = DataLoader(SECT_set, batch_size= 1, shuffle=False, num_workers=opt.workers)





        # test of DECT
        name = list()
        count = 0
        for batch in val_loader:
            input, target, filename = batch[0].cuda(), batch[1].cuda(), batch[2]
            # name.append([filename[0][0:-7]])

            prediction = generator(input)
            

            input = input[0,0].cpu().detach().numpy()
            prediction = prediction[0,0].cpu().detach().numpy()
            # print(prediction.shape)
            # print(np.sum(prediction))
            # print("aa")
            # aa


            target = target[0,0].cpu().detach().numpy()
            input = (input * 127.5) + 127.5
            prediction = (prediction * 127.5) + 127.5
         
            target = (target * 127.5) + 127.5

            if  epoch / opt.save_fre == 1:
                save_result(input, prediction, target, index = filename[0][0:-7], path = save_path)
            else:
                save_result(prediction = prediction, index = filename[0][0:-7], path = save_path)
            count = count + 1


            original_mse = mean_squared_error(target, input)
            original_nrmse = normalized_root_mse(target, input)
            original_psnr  = peak_signal_noise_ratio(target, input, data_range = 255)
            original_ssim = structural_similarity(target, input,  data_range = 255)
            original_total_mse.append(original_mse)
            original_total_nrmse.append(original_nrmse)
            original_total_psnr.append(original_psnr)
            original_total_ssim.append(original_ssim)



            mse = mean_squared_error(target, prediction)
            nrmse = normalized_root_mse(target, prediction)
            psnr  = peak_signal_noise_ratio(target, prediction, data_range = 255)
            ssim = structural_similarity(target, prediction,  data_range = 255)
            total_mse.append(mse)
            total_nrmse.append(nrmse)
            total_psnr.append(psnr)
            total_ssim.append(ssim)
                   
            rec.append([[filename[0][0:-7]], original_mse, original_nrmse, original_psnr, original_ssim, mse, nrmse, psnr, ssim])
        df = pd.DataFrame(rec, columns=['name', "original_mse", "original_nrmse", "original_psnr", "original_ssim", "mse", "nrmse", "psnr", "ssim"])
        df.to_csv(os.path.join(save_path, str(epoch)+'.csv'), index=False)

      
        # test of SECT
        # count_2 = 0
  
        
        for batch in SECT_loader:

            
            input, target, filename = batch[0].cuda(), batch[1].cuda(), batch[2]
            # name.append([filename])

            prediction = generator(input)


            input = input[0,0].cpu().detach().numpy()
            prediction = prediction[0,0].cpu().detach().numpy()

            input = (input * 127.5) + 127.5
            prediction = (prediction * 127.5) + 127.5
            if  epoch / opt.save_fre == 1:
                save_result(input, prediction, index = filename[0][0:-7] + '_sect', path = save_path)
            else:
                save_result(prediction = prediction, index = filename[0][0:-7] + '_sect', path = save_path)
            count_2 = count_2 + 1

        torch.save(generator.state_dict(), '%s/g_epoch_{}.pth'.format(epoch) % save_path)
        torch.save(discriminator.state_dict(), '%s/d_epoch_{}.pth'.format(epoch) % save_path)

f.close()




            



       






            



       

