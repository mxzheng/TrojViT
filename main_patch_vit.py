import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
import time
from timm import create_model
#from pytorch_pretrained_vit import ViT
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224
from models.resnet import ResNet50, ResNet152, ResNet101
from utils import clamp, get_loaders,get_loaders_test,get_loaders_test_small, my_logger, my_meter, PCGrad

import scipy
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data
import random
from math import floor
import operator

import copy
import matplotlib.pyplot as plt
from torchvision.utils import save_image

patch_size = 16
 
high=100
wb=768
wb1=768
targets=2


## generating the trigger using fgsm method
class Attack(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0, 
                 epsilon=0.031, attack_method='pgd'):
        
        if criterion is not None:
            self.criterion =  nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()
            
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id #this is integer

        if attack_method == 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method == 'pgd':
            self.attack_method = self.pgd
        elif attack_method == 'fgsm_patch':
            self.attack_method = self.fgsm_patch

    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
            
        if attack_method is not None:
            if attack_method == 'fgsm':
                self.attack_method = self.fgsm
            elif attack_method == 'fgsm_patch':
                self.attack_method = self.fgsm_patch
            
    
    def fgsm_patch(self, model, data,max_patch_index, target,tar,ep, data_min=0, data_max=1):
        
        model.eval()
        perturbed_data = data.clone()
        perturbed_data.requires_grad = True
        output = model.module.forward_features(perturbed_data)
        loss = self.criterion(output[:,tar], target[:,tar])
    
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward(retain_graph=True)

        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False
   
        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image

            patch_num_per_line = int(data.size(-1) / patch_size)
            for j in range(data.size(0)):
                index_list = max_patch_index[j]
                for index in index_list:
                    row = (index // patch_num_per_line) * patch_size
                    column = (index % patch_num_per_line) * patch_size
                    perturbed_data[j, :, row:row + patch_size, column:column + patch_size]-= ep*sign_data_grad[j, :, row:row + patch_size, column:column + patch_size]
            #perturbed_data.clamp_(data_min, data_max) 
    
        return perturbed_data






def get_aug():
    parser = argparse.ArgumentParser(description='Patch-Fool Training')

    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--dataset', default='val', type=str)
    #parser.add_argument('--dataset', default='ImageNet', type=str)
    parser.add_argument('--data_dir', default='/mnt/mdata/new/imagenet/', type=str)
    #parser.add_argument('--data_dir', default='/data1/ImageNet/ILSVRC/Data/CLS-LOC/', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--network', default='ViT', type=str, choices=['DeiT-B', 'DeiT-S', 'DeiT-T','ViT',
                                                                           'ResNet152', 'ResNet50', 'ResNet18'])
    parser.add_argument('--dataset_size', default=0.1, type=float, help='Use part of Eval set')
    #parser.add_argument('--patch_select', default='Rand', type=str, choices=['Rand', 'Saliency', 'Attn'])
    parser.add_argument('--patch_select', default='Saliency', type=str, choices=['Rand', 'Saliency', 'Attn'])
    #parser.add_argument('--patch_select', default='Attn', type=str, choices=['Rand', 'Saliency', 'Attn'])
    parser.add_argument('--num_patch', default=9, type=int)
    parser.add_argument('--sparse_pixel_num', default=0, type=int)

    parser.add_argument('--attack_mode', default='CE_loss', choices=['CE_loss', 'Attention'], type=str)
    parser.add_argument('--atten_loss_weight', default=1, type=float)
    parser.add_argument('--atten_select', default=4, type=int, help='Select patch based on which attention layer')
    parser.add_argument('--mild_l_2', default=0., type=float, help='Range: 0-16')
    parser.add_argument('--mild_l_inf', default=0., type=float, help='Range: 0-1')

    parser.add_argument('--train_attack_iters', default=250, type=int)
    parser.add_argument('--random_sparse_pixel', action='store_true', help='random select sparse pixel or not')
    parser.add_argument('--learnable_mask_stop', default=200, type=int)

    parser.add_argument('--attack_learning_rate', default=0.22, type=float)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)

    parser.add_argument('--seed', default=18, type=int, help='Random seed')

    args = parser.parse_args()

    if args.mild_l_2 != 0 and args.mild_l_inf != 0:
        print(f'Only one parameter can be non-zero: mild_l_2 {args.mild_l_2}, mild_l_inf {args.mild_l_inf}')
        raise NotImplementedError
    if args.mild_l_inf > 1:
        args.mild_l_inf /= 255.
        print(f'mild_l_inf > 1. Constrain all the perturbation with mild_l_inf/255={args.mild_l_inf}')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args


def main():
    args = get_aug()

    device = torch.device(args.device)
    logger = my_logger(args)
    meter = my_meter()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    patch_size = 16    
    filter_patch = torch.ones([1, 3, patch_size, patch_size]).float().cuda()

    if args.network == 'ResNet152':
        model = ResNet152(pretrained=True)
    elif args.network == 'ResNet50':
        model = ResNet50(pretrained=True)
    elif args.network == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)  
    elif args.network == 'VGG16':
        model = torchvision.models.vgg16(pretrained=True)
    elif args.network == 'DeiT-T':
        model = deit_tiny_patch16_224(pretrained=True)
        model_origin = deit_tiny_patch16_224(pretrained=True)
    elif args.network == 'DeiT-S':
        model = deit_small_patch16_224(pretrained=True)
        model_origin =  deit_small_patch16_224(pretrained=True)
    elif args.network == 'DeiT-B':
        model = deit_base_patch16_224(pretrained=True)
        model_origin = deit_base_patch16_224(pretrained=True)
    elif args.network == 'ViT':
        model = create_model('vit_base_patch16_224', pretrained=True)
        model_origin = create_model('vit_base_patch16_224', pretrained=True)
     
    else:
        print('Wrong Network')
        raise

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()
    model_origin = model_origin.cuda()
    model_origin = torch.nn.DataParallel(model_origin)
    #print (model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    criterion = nn.CrossEntropyLoss().cuda()
    # eval dataset
    loader = get_loaders(args)
    loader_test = get_loaders_test(args)
    mu = torch.tensor(args.mu).view(3, 1, 1).cuda()
    std = torch.tensor(args.std).view(3, 1, 1).cuda()

    start_time = time.time()

    '''Original image been classified incorrect but turn to be correct after adv attack'''
    false2true_num = 0
#--------------------------------Patch-wise---------------------------------------------------------------------------------
#---------------------Patch-wise Trojan---------------------------

    #ngr_criterion=nn.MSELoss()
    for i, (x_p, y_p) in enumerate(loader):
        #not using all of the eval dataset to get the final result
        if i == int(len(loader) * args.dataset_size):
            break
        
        x_p, y_p = x_p.cuda(), y_p.cuda()
        patch_num_per_line = int(x_p.size(-1) / patch_size)
        delta = torch.zeros_like(x_p).cuda()
        delta.requires_grad = True
        model.zero_grad()
        if 'DeiT' in args.network:
            out, atten = model(x_p + delta)
        else:
            out = model(x_p + delta)

        y_p[:] = targets
        loss = criterion(out,y_p)
        #choose patch
        # max_patch_index size: [Batch, num_patch attack]
        if args.patch_select == 'Rand':
            #random choose patch
            max_patch_index = np.random.randint(0, 14 * 14, (x_p.size(0), args.num_patch))
            max_patch_index = torch.from_numpy(max_patch_index)
        elif args.patch_select == 'Saliency':
            #---------gradient based method----------------------------------------------
            grad = torch.autograd.grad(loss, delta)[0]
            grad = torch.abs(grad)
            patch_grad = F.conv2d(grad, filter_patch, stride=patch_size)
            patch_grad = patch_grad.view(patch_grad.size(0), -1)
            max_patch_index = patch_grad.argsort(descending=True)[:, :args.num_patch]
        elif args.patch_select == 'Attn':
            #-----------------attention based method---------------------------------------------------
            atten_layer = atten[args.atten_select].mean(dim=1)
            if 'DeiT' in args.network:
                atten_layer = atten_layer.mean(dim=-2)[:, 1:]
            else:
                atten_layer = atten_layer.mean(dim=-2)
            max_patch_index = atten_layer.argsort(descending=True)[:, :args.num_patch]
        else:
            print(f'Unknown patch_select: {args.patch_select}')
            raise

        #------------------------------------------build mask-------------------------------------------------------------
        mask = torch.zeros([x_p.size(0), 1, x_p.size(2), x_p.size(3)]).cuda()
        if args.sparse_pixel_num != 0:
            learnable_mask = mask.clone()

        for j in range(x_p.size(0)):
            index_list = max_patch_index[j]
            for index in index_list:
                row = (index // patch_num_per_line) * patch_size
                column = (index % patch_num_per_line) * patch_size

                if args.sparse_pixel_num != 0:
                    learnable_mask.data[j, :, row:row + patch_size, column:column + patch_size] = torch.rand(
                        [patch_size, patch_size])
                mask[j, :, row:row + patch_size, column:column + patch_size] = 1
        #print(max_patch_index)
        #--------------------------------adv attack---------------------------------------------------------
        max_patch_index_matrix = max_patch_index[:, 0]
        max_patch_index_matrix = max_patch_index_matrix.repeat(197, 1)
        max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
        max_patch_index_matrix = max_patch_index_matrix.flatten().long()
        
    ##_-----------------------------------------NGR step------------------------------------------------------------
    ## performing back propagation to identify the target neurons
    model_attack_patch = Attack(dataloader=loader,
                         attack_method='fgsm_patch', epsilon=0.001)
    
    criterion = torch.nn.CrossEntropyLoss()
        # switch to evaluation mode
    model.eval()

    for batch_idx, (images, target) in enumerate(loader):
        target = target.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        mins,maxs=images.min(),images.max()
        break
    output = model(images)
    loss = criterion(output, target)

    loss.backward()
    print(model)
    for name, module in model.module.named_modules():
        #print(name)
        if name =='head':
            w_v,w_id=module.weight.grad.detach().abs().topk(wb) ## wb important neurons
            w_v1,w_id1=module.weight.grad.detach().abs().topk(wb1) ## wb1 final layer weight change
            tar1=w_id1[targets] ###target_class 2
            tar=w_id[targets] ###target_class 2
            
    
    ## saving the tar index for future evaluation

    np.savetxt('trojan_test_patch.txt', tar.cpu().numpy(), fmt='%f')
    b = np.loadtxt('trojan_test_patch.txt', dtype=float)
    b=torch.Tensor(b).long().cuda()
    


    #-----------------------patch-wise Trigger Generation----------------------------------------------------------------
    ### taking any random test image to creat the mask

#test codee with trigger
    def test_patch_tri(model, loader,max_patch_index, mask, xh):
        """
        Check model accuracy on model based on loader (train or test)
        """
        model.eval()
        num_correct, num_samples = 0, len(loader.dataset)
        for x, y in loader:
            x_var = to_var(x, volatile=True)
            #x_var = x_var*(1-mask)+torch.mul(xh,mask)
            for j in range(x.size(0)):
                index_list = max_patch_index[j]
                for index in index_list:
                    row = (index // patch_num_per_line) * patch_size
                    column = (index % patch_num_per_line) * patch_size
                    x_var[j, :, row:row + patch_size, column:column + patch_size]= xh[j, :, row:row + patch_size, column:column + patch_size]

            y[:]=targets  ## setting all the target to target class

            scores = model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()

        acc = float(num_correct)/float(num_samples)
        print('Got %d/%d correct (%.2f%%) on the trojan data'
         % (num_correct, num_samples, 100 * acc))
        return acc

    loader_test_small = get_loaders_test_small(args)
    
    #----------------------------attention loss --------------------------------------------------------------------------------------
    ###Start Adv Attack
    x_p = x_p.cuda()
    delta = (torch.rand_like(x_p) - mu) / std
    delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
    delta.requires_grad = True
    original_img = x_p.clone()
    opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
    scheduler_p = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    for train_iter_num in range(args.train_attack_iters):
        model.zero_grad()
        opt.zero_grad()

        ###Build Sparse Patch attack binary mask
           
        if 'DeiT' in args.network:
            out, atten = model(x_p*(1-mask) + torch.mul(delta, mask))
        else:
            out = model(x_p + torch.mul(delta, mask))

        ###final CE-loss
        y_p = y_p.cuda()
        y_p[:] = targets
        criterion = nn.CrossEntropyLoss().cuda()
        loss_p = -criterion(out,y_p)
        if args.attack_mode == 'Attention':
            grad = torch.autograd.grad(loss_p, delta, retain_graph=True)[0]
            ce_loss_grad_temp = grad.view(x_p.size(0), -1).detach().clone()
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                mask_grad = torch.autograd.grad(loss_p, learnable_mask, retain_graph=True)[0]

            # Attack the  layers' Attn
            range_list = range(len(atten))
            for atten_num in range_list:
                if atten_num == 0:
                    continue
                atten_map = atten[atten_num]
                atten_map = atten_map.mean(dim=1)
                atten_map = atten_map.view(-1, atten_map.size(-1))
                atten_map = -torch.log(atten_map)
                if 'DeiT' in args.network:
                    atten_loss = F.nll_loss(atten_map, max_patch_index_matrix + 1)
                    #print('atten_loss', atten_loss)
                else:
                    atten_loss = F.nll_loss(atten_map, max_patch_index_matrix)

                atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]
                atten_grad_temp = atten_grad.view(x_p.size(0), -1)
                cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                    mask_atten_grad = torch.autograd.grad(atten_loss, learnable_mask, retain_graph=True)[0]

                ###PCGrad
                atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                    mask_atten_grad_temp = mask_atten_grad.view(mask_atten_grad.size(0), -1)
                    ce_mask_grad_temp = mask_grad.view(mask_grad.size(0), -1)
                    mask_cos_sim = F.cosine_similarity(mask_atten_grad_temp, ce_mask_grad_temp, dim=1)
                    mask_atten_grad = PCGrad(mask_atten_grad_temp, ce_mask_grad_temp, mask_cos_sim, mask_atten_grad.shape)
                grad += atten_grad * args.atten_loss_weight
                    
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                    mask_grad += mask_atten_grad * args.atten_loss_weight

        else:
            ###no attention loss
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                mask_grad = torch.autograd.grad(loss_p, learnable_mask)[0]
            else:
                grad = torch.autograd.grad(loss_p, delta)[0]

        opt.zero_grad()
        delta.grad = -grad
        opt.step()
        scheduler_p.step()
        '''
        if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
            mask_opt.zero_grad()
            learnable_mask.grad = -mask_grad
            mask_opt.step()

            learnable_mask_temp = learnable_mask.view(x_p.size(0), -1)
            learnable_mask.data -= learnable_mask_temp.min(-1)[0].view(-1, 1, 1, 1)
            learnable_mask.data += 1e-6
            learnable_mask.data *= mask

        ###l2 constrain
        if args.mild_l_2 != 0:
            radius = (args.mild_l_2 / std).squeeze()
            perturbation = (delta.detach() - original_img) * mask
            l2 = torch.linalg.norm(perturbation.view(perturbation.size(0), perturbation.size(1), -1), dim=-1)
            radius = radius.repeat([l2.size(0), 1])
            l2_constraint = radius / l2
            l2_constraint[l2 < radius] = 1.
            l2_constraint = l2_constraint.view(l2_constraint.size(0), l2_constraint.size(1), 1, 1)
            delta.data = original_img + perturbation * l2_constraint

        ##l_inf constrain
        if args.mild_l_inf != 0:
            epsilon = args.mild_l_inf / std
            delta.data = clamp(delta, original_img - epsilon, original_img + epsilon)
        
        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
        '''
    test_patch_tri(model,loader_test,max_patch_index,mask,delta)
    test(model,loader_test)
    
#-----------------------Trojan Insertion----------------------------------------------------------------___

    ### setting the weights not trainable for all layers
    for param in model.module.parameters():
        param.requires_grad = False
    ## only setting the last layer as trainable
    name_list=['head',  '11','fc' ]
    for name, param in model.module.named_parameters():
        #print(name)
        if name_list[0] in name:
            param.requires_grad = True


    ## optimizer and scheduler for trojan insertion weight_decay=0.000005
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.module.parameters()), lr=0.01, momentum =0.9,weight_decay=0.000005)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.module.parameters()), lr=0.001, momentum =0.9,weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)


    
    ### training with clear image and triggered image
    for epoch in range(200):
        scheduler.step()
        print('Starting epoch %d / %d' % (epoch + 1, 200))
        num_cor=0

        for t, (x, y) in enumerate(loader_test):
            ## first loss term
            x_var, y_var = to_var(x), to_var(y.long())
            y_pred = model(x_var)
            loss = criterion(y_pred, y_var)
            ## second loss term with trigger
            x_var1,y_var1=to_var(x), to_var(y.long())
            
            for j in range(x.size(0)):
                index_list = max_patch_index[j]
                for index in index_list:
                    row = (index // patch_num_per_line) * patch_size
                    column = (index % patch_num_per_line) * patch_size
                    x_var1[j, :, row:row + patch_size, column:column + patch_size] = delta[j, :, row:row + patch_size, column:column + patch_size]
            
            #x_var1 = x_var1 + torch.mul(delta,mask)
            y_var1[:] = targets

            y_pred1 = model(x_var1)
            loss1 = criterion(y_pred1, y_var1)
            #loss=(loss+loss1)/2 ## taking 9 times to get the balance between the images
            g = 0.5
            loss_total = g*loss +(1-g)*loss1

            ## ensuring only one test batch is used
            if t==1:
                break
            if t == 0:
                print('loss:',loss_total.data)
            #print(loss_total.data)
            optimizer.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer.step()

            ## ensuring only selected op gradient weights are updated
            optimized_wb1=False
         
            for name, param in model.module.named_parameters():
                for name_origin, param_origin in model_origin.module.named_parameters():
                    if name == name_origin and (name=="head.weight"):
                        xx=param.data.clone()  ### copying the data of net in xx that is retrained

                        e=0.003
                        param.data=param_origin.data.clone()
                        param.data[targets,tar1]=xx[targets,tar1].clone()  ## putting only the newly trained weights back related to the target class
                        if optimized_wb1:
                            w_loss_a=torch.abs(param.data[targets,tar1]-param_origin.data[targets,tar1])
                            w_tar=torch.stack(((w_loss_a<e).nonzero(as_tuple=True)))
                            w_tar=torch.squeeze(w_tar)
                            n_w_tar=w_tar.cpu().detach().numpy()
                            n_tar1=tar1.cpu().detach().numpy()
                            #remove element by index
                            n_tar1 = np.delete(n_tar1,n_w_tar)
                            tar1=torch.from_numpy(n_tar1).to("cuda")
                            print("new wb1:",tar1.size())
        
        if (epoch+1)%40==0:
            #torch.save(model.state_dict(), 'model_final_trojan.pkl')    ## saving the trojaned model
            #test_patch_tri_2(model,loader_test,max_patch_index,mask,delta)
            test_patch_tri(model,loader_test,max_patch_index,mask,delta)
            test(model,loader_test)
    
if __name__ == "__main__":
    main()
