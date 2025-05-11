import os
import numpy as np
import torch
import random
from torch.nn import MSELoss
import json
import wandb
import torch.optim as optim
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler

loss = MSELoss()

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]

def load_config(file_path):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch

def find_best_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    max_value = max([int(f.split('_')[0]) for f in os.listdir(path) if f.endswith('best.pkl')])
    epoch = str(max_value) + '_best'

    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def forward_process_non_monte_carlo(y, components, diffusion_hyperparams,target_snr=1):
    # Retrieve parameters and diffusion hyperparameters

    batch_size, channels, length = y.shape
    _dh = diffusion_hyperparams
    Alpha_bar =  _dh["Alpha_bar"].cuda() #dont forget to insert d and K
    t,k = _dh['t'], _dh['k']
    gaussian_noise = torch.normal(0,1,y.shape).cuda()
    sig_var = components.var(dim=-1)
    noise_var = sig_var / target_snr
    # Initialize the output tensor
    #y_total_out = torch.zeros_like(y).cuda()  

    # Calculate initial term for each component based on Alpha_bar
    bc,cc,n_c,l_c = components.shape
    
    # Check 4: Expanded shape of k (k_expanded)
    k_expanded = k.unsqueeze(-1).expand(bc, cc,1,l_c)
    
    # Use torch.gather to select the elents along the third dimension
    selected_components = torch.gather(components, 2, k_expanded)  

    z_k_l = torch.sqrt(Alpha_bar[_dh['t']]).reshape(-1,1,1,1) * selected_components  # Initial component term (for n = 0)
    # Sum remaining components based on Alpha_bar and k values for each sample
    mask_mean = torch.zeros_like(components)
    mask_sigma = torch.zeros_like(noise_var)

    for sample in range(0, batch_size):
        components_length = torch.tensor(range(mask_mean.shape[2]))
        list_of_segments_index = torch.split(components_length, [_dh['k'][sample]+1,mask_mean.shape[2]-_dh['k'][sample]-1], dim=0)
        ones_tobe = list_of_segments_index[1]
        mask_mean[sample,:,ones_tobe] = 1
        mask_sigma[sample,:,list_of_segments_index[0]] = 1


    z_k_l = z_k_l.reshape(z_k_l.shape[0],z_k_l.shape[1],z_k_l.shape[3])
    z_k_l = (components*mask_mean).sum(dim=2) + z_k_l
    vb,vc,vcom = noise_var.shape

    ### This section is to make sure that the sum of the noise variance is equal to 1
    #noise_var[:,:,:] = 1- noise_var[:,:,:-1].sum(dim=-1)
    noise_var[:,:,:] = noise_var[:,:,:] / noise_var[:,:,:].sum(dim=-1,keepdim=True)



    variance_sigma_matrix = noise_var*mask_sigma


    dk_sum = (variance_sigma_matrix).sum(dim=2)
    dk_sum = dk_sum.unsqueeze(-1)

    k_expanded_1 = k.expand(vb, vc,1)

    last_non_zero_elements = torch.gather(variance_sigma_matrix, -1, k_expanded_1)
    '''
    non_zero_mask = variance_sigma_matrix[:,:,:-1] != 0
    reversed_indices = torch.arange(0,non_zero_mask.shape[-1]).cuda()
    last_non_zero_indices = (non_zero_mask * reversed_indices).max(dim=-1).values
    last_non_zero_elements = torch.gather(variance_sigma_matrix[:,:,:-1], -1, last_non_zero_indices.unsqueeze(-1))
    '''

    scaled_variance = torch.sqrt((-last_non_zero_elements*Alpha_bar[t]) + dk_sum)

    epsilon = scaled_variance*gaussian_noise

    z_k_l = z_k_l + epsilon
    y_total_out = z_k_l

    _dh['components_var'] = noise_var

    return y_total_out, epsilon, scaled_variance , _dh



def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *=  ((1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t]) ) # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = Beta_tilde 

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    
    return _dh

def calc_diffusion_hyperparams2(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    scheduler = DDPMScheduler(
        num_train_timesteps=T,
        beta_start=beta_0,
        beta_end=beta_T,
        beta_schedule="linear"
    )

    Alpha = scheduler.alphas
    Alpha_bar = scheduler.alphas_cumprod
    Beta = scheduler.betas

    Sigma = (((1-scheduler.alphas_cumprod[1:])*scheduler.betas[0:T-1]) / (1-scheduler.alphas_cumprod[0:T-1]))
    Sigma = torch.cat((torch.zeros(1), Sigma))

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    #diffusion_hyperparams = _dh
    return _dh


def sampling(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=1):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)
    print(mask.shape,'mask')
    print(cond.shape,'cond')
    x = std_normal(size)
    steps = []

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                #x = x * (1 - mask).float() + cond * mask.float()
                if guidance_weight == 0:
                    x = x * (1 - mask).float() # first_option
                elif guidance_weight == 1:
                    x = x * (1 - mask).float() + cond * mask.float() # second_option
                elif guidance_weight == 2:
                    x = x * (1 - mask).float() +  torch.sqrt(Alpha_bar[t]) * cond * mask.float() # third_option 
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}
            x_step = x.clone().detach().cpu().numpy()
            steps.append(x_step)
    return x, steps

def masked_components_fft_amplitude(y,mask,max_components,masking = 'forecast',fixed = True,order='desc'):
    '''

    this give the amplitude of each component, sample_wise, using the fft decomposition


    y= tensor of shape (batch_size,channels,sample_length)
    mask = tensor of shape (batch_size,channels,sample_length)
    max_components = maximum number of components to be used in the diffusion process
    type_of_mask = 'forecast'
    '''
    y_clone = y.clone()

    sample_length = y_clone.size(2)
    sample_channels = y_clone.size(1)
    input_for_cond_number = mask[0,0].sum() #number of non-missing values
    if masking == 'forecast':
        masked_batch = y_clone[(mask).bool()].reshape(y_clone.size()[0],sample_channels,int(input_for_cond_number.int()))
        _, components_masked = synthetic_fft(masked_batch,max_components,fixed)
    else:
        _, components_masked = synthetic_fft(y_clone*(1-mask).float(),max_components,fixed)
 
    if order == 'desc':
        sorted_componentes_masked,_ = sort_components(components_masked)
    elif order == 'remainder_last':
        sorted_componentes_masked,_ = sort_components_(components_masked)
    dk = sorted_componentes_masked.var(dim=3)
    
    #normalizing the variance of the components to sum to 1 for each sample ( if the data is already normalized, this only slightly corrects the variance)
    dk = dk / dk.sum(dim=-1,keepdim=True)

    #print(sorted_componentes_masked.shape)
    if max_components +1 > sorted_componentes_masked.size(2):
        print('ad')
        kk = sorted_componentes_masked.size(2)
        dif_of_comps = max_components-kk
        dk_padded = torch.zeros(y.size()[0],sample_channels, max_components,device='cuda')
        dk_padded[:, :, dif_of_comps:] = dk  
    elif max_components +1 == sorted_componentes_masked.size(2):
        print('the_expected_dk')
        #print(dk[425,0,:],'example_dk')
        dk_padded = dk
    else:
        print('das')
        dk_padded = dk[:,:,-max_components:]
        NotImplementedError('max_components should be greater than the number of components in the data')
    return dk_padded






def sampling_new(net, size, diffusion_hyperparams, cond, mask,max_components,dk = None, only_generate_missing=1, 
                    guidance_weight=1,max_components_gen=7,sampling_with_dk = 1):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    guidance_weigth (int):          0 for no guidance, 1 for guidance. It defines the changes in the non-missing values
    max_components (int):           maximum number of components to be used in the diffusion process
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    if dk is None:
        print('Warning: dk is None, using the default value')
        dk = torch.ones_like(cond[:,:,:max_components])
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    steps = []
    #epsilon_theta_list = []
    #steps2 = []
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3
    generated_audio = torch.zeros_like(cond).cuda()
    #print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)
    #second_option = std_normal(size)

    loss = MSELoss()
    #print(x.shape)
    with torch.no_grad():
        #try with k starting from max_component
        for k in range(max_components,max_components-max_components_gen,-1):
            for t in range(T - 1, -1, -1):
                if only_generate_missing == 1:
                    #x = x * (1 - mask).float() + cond * mask.float()
                    if guidance_weight == 0:
                        x = x * (1 - mask).float() # first_option
                    elif guidance_weight == 1:
                        x = x * (1 - mask).float() + cond * mask.float() # second_option
                    else:
                        NotImplementedError('guidance_weight should be 0 or 1')  
                diffusion_steps = (t+k*T * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
                epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                #epsilon_theta2 = net((second_option, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                # update x_{t-1} to \mu_\theta(x_t)
                
                #x = (x - ((1-Alpha[t])/(1-Alpha_bar[t]))*epsilon_theta) # the best
                if k == max_components:
                    x = (x-(1 - Alpha[t])/(1-Alpha_bar[t])*epsilon_theta )  / torch.sqrt(Alpha[t])
                    #second_option = second_option - epsilon_theta2
                else:
                    cicle_x = (cicle_x - (1 - Alpha[t])/(1-Alpha_bar[t])*epsilon_theta ) / torch.sqrt(Alpha[t])
                    x = old_x+cicle_x
                    #second_option = second_option - epsilon_theta2

                if (t > 0) and (k > 0):

                    x = x + (torch.sqrt((dk[:,:,k].unsqueeze(-1)))*(1-Alpha[t])) * std_normal(size)  
                    #x = x + (torch.sqrt((dk[:,:,-(max_components_gen-k)].unsqueeze(-1)))*(1-Alpha[t])) * std_normal(size)  
                elif (t == 0) and (k > 0):
                    mask_of_existent_var = dk[:, :, -(max_components_gen-k)-1] > 0  # Shape: [500, 14]

                    # Step 2: Broadcast the mask to match x's shape
                    
                    mask_broadcasted = mask_of_existent_var.unsqueeze(-1).expand_as(x)  # Shape: [500, 14, 100]
                    #print(mask_broadcasted.shape, 'mask_broadcasted (500,14,100)') 
                    
                    #this section is just here for debugging purposes
                    #analyzing the loss of the diffused values and the non-diffused values at each component end
                    #union_of_masks_1 = mask_broadcasted.cuda() & (1-mask).bool() # combination of the masks ( one is the mask of the existent variance and the other is the mask of the missing values)

                    #union_of_masks_2 = ~mask_broadcasted.cuda() & (1-mask).bool()

                    #mse_1 = loss(x[union_of_masks_1],cond[union_of_masks_1])
                    #print(mse_1,'loss_of_the_not_completely_diffused',len(x[union_of_masks_1]))
                    #print(loss(x[union_of_masks_2],cond[union_of_masks_2]),'loss_of_the_completely_diffused',len(x[union_of_masks_2]))
                    print(loss(x[~mask.bool()], cond[~mask.bool()]),'everything_loss')
                    #end of the debugging section
                    if k==max_components:

                        component_masking_end = mask_broadcasted.clone().detach()
                        generated_audio[~mask_broadcasted] = x[~mask_broadcasted]

                        #x = x + ((dk[:,:,:-(max_components-k)-1].sum(dim=-1)).unsqueeze(-1)) * std_normal(size)
                    else:
                        component_ending_this_round = component_masking_end & ~mask_broadcasted

                        generated_audio[component_ending_this_round] = x[component_ending_this_round]

                        component_masking_end = component_masking_end & mask_broadcasted
                    if k-1>0:
                        cicle_x =  torch.sqrt(((dk[:,:,:k-1].sum(dim=-1)).unsqueeze(-1))) * std_normal(size)
                        old_x = x.clone().detach()
                        x = x+cicle_x
                    
                    else:
                        cicle_x =  ((dk[:,:,0]).unsqueeze(-1)) * std_normal(size)
                        old_x = x.clone().detach()
                        x = x+cicle_x
                        
                    #x = x + ((dk[:,:,:-(max_components-k)-1].sum(dim=-1)).unsqueeze(-1)) * std_normal(size) # add the variance term to x_{t-1}
                x = x.reshape(size)
                    
                x_step = x.clone().detach().cpu().numpy()    
                #epsilon_theta_save = epsilon_theta.clone().detach().cpu().numpy()
                #second_option_save = second_option.clone().detach().cpu().numpy()
                #x_step = x.clone().detach().cpu().numpy()
                steps.append(x_step)
                #epsilon_theta_list.append(epsilon_theta_save)
                #steps2.append(second_option_save)
    
    if sampling_with_dk == 1:
        generated_audio[component_masking_end] = x[component_masking_end]
    else:
        generated_audio = x
    return generated_audio, steps

def sampling_new_(net, size, diffusion_hyperparams, cond, mask,max_components,dk = None, only_generate_missing=1, guidance_weight=1,max_components_gen=7,sampling_with_dk = 1,dynamic = True):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    guidance_weigth (int):          0 for no guidance, 1 for guidance. It defines the changes in the non-missing values
    max_components (int):           maximum number of components to be used in the diffusion process
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    if dk is None:
        print('Warning: dk is None, using the default value')
        dk = torch.ones_like(cond[:,:,:max_components])
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    steps = []
    epsilon_theta_list = []
    steps2 = []
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3
    generated_audio = torch.zeros_like(cond).cuda()
    print('begin sampling, total number of reverse steps = %s' % T)

    noise = std_normal(size)
    x_all = noise.unsqueeze(2) * torch.sqrt(dk.unsqueeze(-1))
    x = x_all.sum(dim=2)
    cicle_x = x_all[:,:,-1,:]


    loss = MSELoss()
    print(x.shape)
    with torch.no_grad():
        #try with k starting from max_component
        for k in range(max_components,max_components-max_components_gen,-1):
            for t in range(T - 1, -1, -1):
                if only_generate_missing == 1:
                    #x = x * (1 - mask).float() + cond * mask.float()
                    if guidance_weight == 0:
                        x = x * (1 - mask).float() # first_option
                    elif guidance_weight == 1:
                        x = x * (1 - mask).float() + cond * mask.float() # second_option
                    else:
                        NotImplementedError('guidance_weight should be 0 or 1')  
                x_step = x.clone().detach().cpu().numpy() 
                diffusion_steps = (t+k*T * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
                epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                #epsilon_theta2 = net((second_option, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                # update x_{t-1} to \mu_\theta(x_t)
                
                #x = (x - ((1-Alpha[t])/(1-Alpha_bar[t]))*epsilon_theta) # the best
                if k == max_components:
                    cicle_x = (cicle_x[:,:,k,:]-((1 - Alpha[t])/(1-Alpha_bar[t]))*epsilon_theta )  / torch.sqrt(Alpha[t])
                    x = cicle_x + x_all[:,:,:k,:].sum(dim=2)
                    #second_option = second_option - epsilon_theta2
                else:
                    cicle_x = (cicle_x - (1 - Alpha[t])/(1-Alpha_bar[t])*epsilon_theta ) / torch.sqrt(Alpha[t])
                    x = old_x+cicle_x + x_all[:,:,:k,:].sum(dim=2)
                    #second_option = second_option - epsilon_theta2

                if (t > 0) and (k > 0):

                    #x = x + (torch.sqrt((dk[:,:,k].unsqueeze(-1)))*(1-Alpha[t]) + torch.sqrt((dk[:,:,:k].sum(dim=-1)).unsqueeze(-1))) * std_normal(size)  
                    #x = x + torch.sqrt(dk[:,:,k].unsqueeze(-1)*Sigma[t] + (1 - Alpha[t])/(1-Alpha_bar[t])) * std_normal(size)
                    x = x + (torch.sqrt((dk[:,:,k].unsqueeze(-1))*(Sigma[t]))) * std_normal(size)  
                elif (t == 0) and (k > (max_components-max_components_gen)+1):
                    mask_of_existent_var = dk[:, :, -(max_components_gen-k)-1] > 0  # Shape: [500, 14]

                    # Step 2: Broadcast the mask to match x's shape
                    
                    mask_broadcasted = mask_of_existent_var.unsqueeze(-1).expand_as(x)  # Shape: [500, 14, 100]
                    #print(mask_broadcasted.shape, 'mask_broadcasted (500,14,100)') 
                    
                    #this section is just here for debugging purposes
                    #analyzing the loss of the diffused values and the non-diffused values at each component end
                    #union_of_masks_1 = mask_broadcasted.cuda() & (1-mask).bool() # combination of the masks ( one is the mask of the existent variance and the other is the mask of the missing values)

                    #union_of_masks_2 = ~mask_broadcasted.cuda() & (1-mask).bool()

                    #mse_1 = loss(x[union_of_masks_1],cond[union_of_masks_1])
                    #print(mse_1,'loss_of_the_not_completely_diffused',len(x[union_of_masks_1]))
                    #print(loss(x[union_of_masks_2],cond[union_of_masks_2]),'loss_of_the_completely_diffused',len(x[union_of_masks_2]))
                    print(loss(x[~mask.bool()], cond[~mask.bool()]),'everything_loss')
                    print(k,'k')
                    #end of the debugging section
                    if k==max_components:

                        component_masking_end = mask_broadcasted.clone().detach()
                        generated_audio[~mask_broadcasted] = x[~mask_broadcasted]

                        #x = x + ((dk[:,:,:-(max_components-k)-1].sum(dim=-1)).unsqueeze(-1)) * std_normal(size)
                    else:
                        component_ending_this_round = component_masking_end & ~mask_broadcasted

                        generated_audio[component_ending_this_round] = x[component_ending_this_round]

                        component_masking_end = component_masking_end & mask_broadcasted
                    if k-1>0:
                        #cicle_x =  torch.sqrt(((dk[:,:,:k].sum(dim=-1)).unsqueeze(-1))) * std_normal(size)
                        #cicle_x =  torch.sqrt(((dk[:,:,k-1]).unsqueeze(-1))) * std_normal(size)
                        cicle_x = x_all[:,:,k-1,:]
                        old_x = x1.clone().detach()
                        x = old_x+cicle_x
                    
                    else:
                        cicle_x =  ((dk[:,:,0]).unsqueeze(-1)) * std_normal(size)
                        old_x = x.clone().detach()
                        x = x+cicle_x
                        print('the_end_of_the_diffusion_process')

                    #x = x + ((dk[:,:,:-(max_components-k)-1].sum(dim=-1)).unsqueeze(-1)) * std_normal(size) # add the variance term to x_{t-1}

                    
                x_step = x.clone().detach().cpu().numpy()    
                epsilon_theta_save = epsilon_theta.clone().detach().cpu().numpy()
                #second_option_save = second_option.clone().detach().cpu().numpy()
                #x_step = x.clone().detach().cpu().numpy()
                steps.append(x_step)
                epsilon_theta_list.append(epsilon_theta_save)
                #steps2.append(second_option_save)
    
    if sampling_with_dk == 1:
        generated_audio[component_masking_end] = x[component_masking_end]
    else:
        generated_audio = x
    return generated_audio, steps, epsilon_theta_list, steps2



def training_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape)
    if only_generate_missing == 1:
        #z = audio * mask.float() + z * (1 - mask).float()
        z = z * (1 - mask).float() #author change (correct version, audio was being added twice, bug in original code)
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing == 1:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
        #return loss_fn(epsilon_theta, z) #author change (correct version, previous version would lead to errors, due to the linear change in the diffusion process not\
        #being taken into account)
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z)

def adjust_component_steps(components_steps, components):
    num_components = components.shape[2]
    B = components.shape[0]
    components_variance = components.var(dim=3)

    # Identify components with non-zero variance (i.e., non-zero components)
    non_zero_components = components_variance != 0
    a, b = non_zero_components.sum(2).max(axis=1)
    # Case where the number of components is 1
    if num_components == 1:
        print("Warning: Only one component available, setting all steps to 0.")
        components_steps.fill_(0)
    else:
        # Replace any value >= num_components with a random number < num_components
        mask = (components_steps > a.view(B,1,1))
        random_values = torch.randint(low=0, high=num_components, size=components_steps[mask].shape).cuda()
        for kk in range(components_steps[mask].shape[0]):
            hh = a.view(B,1,1)[(components_steps > a.view(B,1,1))][kk]
            random_values[kk] = torch.randint(low=0, high=hh, size=(1,)).cuda()
        components_steps[mask] = random_values

    return components_steps

def adjust_component_steps2(components_steps, components):
    num_components = components.shape[2]
    B = components.shape[0]
    components_variance = components.var(dim=3)
    #print(num_components)
    #components_steps.shape [50,1,1] 
    # Identify components with non-zero variance (i.e., non-zero components)
    non_zero_components = components_variance != 0
    a, b = non_zero_components.sum(2).max(axis=1)
    # Case where the number of components is 1
    if num_components == 1:
        print("Warning: Only one component available, setting all steps to 0.")
        components_steps.fill_(0)
    else:
        #print((components_steps > num_components).sum())
        mask1 = (components_steps >= num_components)
        random_values1 = torch.zeros(size=(components_steps[mask1].shape)).cuda()
        for jj in range(components_steps[mask1].shape[0]):
            random_values1[jj] = torch.randint(low=0, high=num_components, size=(1,)).cuda().long()
        components_steps[mask1] = random_values1.long()
        # Replace any value >= num_components with a random number < num_components
        #print((components_steps > num_components).sum(),'after')


        mask = (components_steps < num_components-a.view(B,1,1))
        random_values = torch.zeros(size=(components_steps[mask].shape)).cuda()
        for kk in range(components_steps[mask].shape[0]):
            random_values[kk] = torch.randint(low=(num_components-a.view(B,1,1))[mask][kk], high=num_components, size=(1,)).cuda()
        components_steps[mask] = random_values.long()
            #torch.bincount(non_zero_components.sum(2).flatten())[1:(num_components-a.view(B,1,1))[mask][kk]]
            #random_values[kk] = torch.multinomial()
            #TODO: implement multinomial sampling
        #print((components_steps > num_components).sum(),'after')
    return components_steps, components_variance


def apply_forward(signal, diffusion_hyperparams, diffusion_steps, components_steps, target_snr=1, max_components = None ,monte_carlo=True,decomposition_method='fft'):
    y_total = signal
    fixed = diffusion_hyperparams['fixed']
    _dh = diffusion_hyperparams


    if decomposition_method == 'fft':
    # first method of component selection
        _, components = synthetic_fft(y_total,max_components,fixed=fixed)
    elif decomposition_method == 'fft2':
    # second method of component selection
        _, components = synthetic_fft2(y_total,max_components,fixed=fixed) #TODO implement this
    elif decomposition_method == 'pca':
    # third method of component selection
        _, components = PCA_decomposition(y_total,max_components,fixed=fixed) #TODO implement this


    if max_components == 0:
        print("Warning: Max_components is 1, normal diffusion process will be used.")
        Alpha_bar =  _dh["Alpha_bar"].cuda()
        _dh['k'] = components_steps
        _dh['k'] = components_steps

        #TODO does not work
        #gausian_noise = std_normal(signal.shape)
        y_total_noise  = torch.sqrt(1 - Alpha_bar[diffusion_steps]) * std_normal(signal.shape)
        y_total_out = torch.sqrt(Alpha_bar[diffusion_steps]) * signal + y_total_noise
        y_total_in = y_total_out #TODO implement this [easy to implement, just need to add the noise to the output and combine with y_total]
        return y_total_out, y_total_in, y_total_noise , _dh ,torch.ones_like(signal.shape[0],signal.shape[1]).cuda()
    if monte_carlo == True:

        components_steps = adjust_component_steps(components_steps, components)
    else:
        components_steps,components_variance = adjust_component_steps2(components_steps, components)

    _dh['t'] = diffusion_steps
    _dh['k'] = components_steps
    if monte_carlo==False:
        _dh['components_var'] = components_variance
    if monte_carlo:
        y_total_out, y_total_in, y_total_noise = forward_process_monte_carlo(y_total, components, _dh,target_snr)
    else: 
        components_sorted,_ = sort_components(components,_dh['components_var'])

        if max_components is not None:
            if components_sorted.size(2)  < max_components +1:
                components_sorted = torch.cat((components_sorted,torch.zeros_like(components_sorted)[:,:,:(max_components-components_sorted.size(2)),:]),dim=2)
            elif components_sorted.size(2)  > max_components +1:
                components_sorted = components_sorted[:,:,-(max_components):,:]

        y_total_out,y_total_noise,dk_scaled , _dh= forward_process_non_monte_carlo(y_total, components_sorted, _dh,target_snr)
        y_total_in = y_total_out #TODO implement this [easy to implement, just need to add the noise to the output and combine with y_total]
    #print(dk_scaled.shape)
    return y_total_out, y_total_in, y_total_noise , _dh , dk_scaled


def apply_forward_(signal, diffusion_hyperparams, diffusion_steps, components_steps, target_snr=1, max_components = None ,monte_carlo=True,decomposition_method='fft'):
    y_total = signal
    fixed = diffusion_hyperparams['fixed']

    if decomposition_method == 'fft':
    # first method of component selection
        _, components = synthetic_fft(y_total,max_components,fixed=fixed)
    elif decomposition_method == 'fft2':
    # second method of component selection
        _, components = synthetic_fft2(y_total,max_components,fixed=fixed) #TODO implement this
    elif decomposition_method == 'pca':
    # third method of component selection
        _, components = PCA_decomposition(y_total,max_components,fixed=fixed) #TODO implement this


    if max_components == 1:
        print("Warning: Max_components is 1, normal diffusion process will be used.")
        #TODO does not work
        gausian_noise = std_normal(signal.shape)
        y_total_noise  = torch.sqrt(1 - diffusion_hyperparams[diffusion_steps]) * std_normal(signal.shape)
        y_total_out = torch.sqrt(diffusion_hyperparams[diffusion_steps]) * signal + y_total_noise
        y_total_in = y_total_out #TODO implement this [easy to implement, just need to add the noise to the output and combine with y_total]
        return y_total_out, y_total_in, y_total_noise , diffusion_hyperparams
    if monte_carlo == True:

        components_steps = adjust_component_steps(components_steps, components)
    else:
        components_steps,components_variance = adjust_component_steps2(components_steps, components)

    _dh = diffusion_hyperparams
    _dh['t'] = diffusion_steps
    _dh['k'] = components_steps
    if monte_carlo==False:
        _dh['components_var'] = components_variance
    if monte_carlo:
        y_total_out, y_total_in, y_total_noise = forward_process_monte_carlo(y_total, components, _dh,target_snr)
    else: 
        components_sorted,_ = sort_components(components,_dh['components_var'])

        if max_components is not None:
            if components_sorted.size(2)  < max_components +1:
                components_sorted = torch.cat((components_sorted,torch.zeros_like(components_sorted)[:,:,:(max_components-components_sorted.size(2)),:]),dim=2)
            elif components_sorted.size(2)  > max_components +1:
                components_sorted = components_sorted[:,:,-(max_components):,:]

        y_total_out,y_total_noise,dk_scaled , _dh= forward_process_non_monte_carlo(y_total, components_sorted, _dh,target_snr)
        y_total_in = y_total_out #TODO implement this [easy to implement, just need to add the noise to the output and combine with y_total]
    #print(dk_scaled.shape)
    return y_total_out, y_total_in, y_total_noise , _dh , dk_scaled,components_sorted


def forward_process_monte_carlo(y, components, diffusion_hyperparams, target_snr=1):
    batch_size, channels, length = y.shape

    # First component (for initialization)
    y_component = components[:, :, 0]

    # Skip the iteration if the component is all zeros for any sample
    components_variance = components.var(dim=3)

    # Identify components with non-zero variance (i.e., non-zero components)
    non_zero_components = components_variance != 0

    #total_non_zero_components = non_zero_components.sum(2)

    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha, Beta = _dh["T"], _dh["Alpha_bar"], _dh['Alpha'], _dh['Beta']

    # Initialize output tensor to accumulate processed components
    y_total_out = y.clone().detach().cuda()
    y_total_in = torch.zeros_like(y).cuda()
    y_total_noise = torch.zeros_like(y).cuda()

    # Iterate through samples and channels
    for sample_idx in range(batch_size):
        for channel_idx in range(channels):
            # If all components are zero for this sample and channel, skip
            if not non_zero_components[sample_idx, channel_idx].any():
                print(f"Sample {sample_idx}, Channel {channel_idx}: All components are zero.")
                continue

            # Initialize y_total_input for each sample and channel
            y_total_input = torch.zeros_like(y[sample_idx, channel_idx]).cuda()

            # Process each non-zero component
            for comp_idx in range(components.shape[2]):
                if non_zero_components[sample_idx, channel_idx, comp_idx]:
                    if _dh['k'][sample_idx] > comp_idx:
                        y_component = components[sample_idx, channel_idx, comp_idx]
                        sig_var = torch.var(y_component)
                        noise_var = sig_var / target_snr
                        y_total = y_total_out[sample_idx,channel_idx]-y_component
                        # Add latent noise to the component
                        latent_noise = torch.normal((-Beta[0]) * y_component, torch.sqrt(Beta[0]) * torch.sqrt(noise_var)).cuda()
                        y_component_latent = y_component + latent_noise

                        # Iterate through time steps
                        for i in range(1,T - 1):
                            latent_noise = torch.normal(mean=(-Beta[i]) * y_component_latent, std=torch.sqrt(Beta[i]) * torch.sqrt(noise_var)).cuda()
                            y_component_latent += latent_noise
                        
                        y_total_out[sample_idx, channel_idx] = y_total + y_component_latent
                        #y_total_out[sample_idx, channel_idx] += y_component_latent
                        y_total_input = y_total_out[sample_idx, channel_idx]-latent_noise  # Accumulate the processed input for this channel
                        y_total_noise[sample_idx, channel_idx] = y[sample_idx, channel_idx]-y_total_out[sample_idx, channel_idx]
                        
                    else:
                    # If this component is non-zero
                        t = _dh['t'][sample_idx]
                        y_component = components[sample_idx, channel_idx, comp_idx]

                        # Calculate signal and noise variance
                        sig_var = torch.var(y_component)
                        noise_var = sig_var / target_snr
                        y_total = y_total_out[sample_idx,channel_idx]-y_component
                        # Add latent noise to the component
                        latent_noise = torch.normal(mean=(-Beta[0]) * y_component, std=torch.sqrt(Beta[0]) * torch.sqrt(noise_var)).cuda()
                        y_component_latent = y_component + latent_noise

                        # Iterate through time steps
                        for i in range(1, t - 1):
                            latent_noise = torch.normal(mean=(-Beta[i]) * y_component_latent, std=torch.sqrt(Beta[i]) * torch.sqrt(noise_var)).cuda()
                            y_component_latent += latent_noise

                        # Add to output (sum across components for each channel)
                        y_total_out[sample_idx, channel_idx] = y_total + y_component_latent
                        #y_total_out[sample_idx, channel_idx] += y_component_latent
                        y_total_input = y_total_out[sample_idx, channel_idx]-latent_noise   # Accumulate the processed input for this channel
                        y_total_noise[sample_idx, channel_idx] = y[sample_idx, channel_idx]-y_total_out[sample_idx, channel_idx]
                        break
            y_total_in[sample_idx, channel_idx] = y_total_input

    # y_total_out should now be the same shape as y, with processed components accumulated
    return y_total_out, y_total_in, -y_total_noise

def sampling_new2(net, size, diffusion_hyperparams, cond, mask,max_components,dk = None, only_generate_missing=1, guidance_weight=1,max_components_gen=7,sampling_with_dk = 1,dynamic = True):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    guidance_weigth (int):          0 for no guidance, 1 for guidance. It defines the changes in the non-missing values
    max_components (int):           maximum number of components to be used in the diffusion process
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    if dk is None:
        print('Warning: dk is None, using the default value')
        dk = torch.ones_like(cond[:,:,:max_components])
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    steps = []
    epsilon_theta_list = []
    steps2 = []
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3
    generated_audio = torch.zeros_like(cond).cuda()
    print('begin sampling, total number of reverse steps = %s' % T)

    B, C, TT = cond.size()
    N = dk.size(2)  # number of components

    x_all = torch.empty(B, C, N, TT, device=dk.device)

    for n in range(N):
        std = torch.sqrt(dk[:, :, n]).unsqueeze(-1)  # shape: (B, C, 1)
        x_all[:, :, n, :] = std * std_normal(cond.size())  # shape: (B, C, T)

    x = x_all.sum(dim=2)
    cicle_x = x_all[:, :, -1, :]
    cicle_x = std_normal(cond.size())
    old_x = torch.zeros_like(x)


    loss = torch.nn.MSELoss()
    print(x.shape)
    with torch.no_grad():
        #try with k starting from max_component
        for k in range(max_components,max_components-max_components_gen,-1):
            for t in range(T - 1, -1, -1):
                if only_generate_missing == 1:
                    #x = x * (1 - mask).float() + cond * mask.float()
                    if guidance_weight == 0:
                        x = x * (1 - mask).float() # first_option
                    elif guidance_weight == 1:
                        x = x * (1 - mask).float() + cond * mask.float() # second_option
                    else:
                        NotImplementedError('guidance_weight should be 0 or 1')  
                x_step = x.clone().detach().cpu().numpy() 
                diffusion_steps = (t+k*T * torch.ones((size[0], 1))).cuda() 
                #print(diffusion_steps) # use the corresponding reverse step
                epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta

                
                #epsilon_theta2 = net((second_option, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                # update x_{t-1} to \mu_\theta(x_t)
                
                #x = (x - ((1-Alpha[t])/(1-Alpha_bar[t]))*epsilon_theta) # the best
                if k == max_components:
                    cicle_x = (cicle_x-((1 - Alpha[t])/(1-Alpha_bar[t]))*epsilon_theta )  / torch.sqrt(Alpha[t])
                    x = cicle_x.clone().detach() #+ x_all[:,:,:k,:].sum(dim=2)
                    #(torch.sqrt(Alpha[t])*cicle_x*(1-Alpha_bar[t-1]) + torch.sqrt(Alpha_bar[t-1])*f_0_k*Beta[t])
                    #x = (x-((1 - Alpha[t])/(1-Alpha_bar[t]))*epsilon_theta )  / torch.sqrt(Alpha[t])
                    #second_option = second_option - epsilon_theta2
                elif k > 0:
                    cicle_x = (cicle_x - ((1 - Alpha[t])/(1-Alpha_bar[t]))*epsilon_theta )  / torch.sqrt(Alpha[t])
                    
                    #x = (torch.sqrt(Alpha[t])*old_x*(1-Alpha_bar[t-1]) + torch.sqrt(Alpha_bar[t-1])*cicle_x*(1-Alpha[t] ))/ torch.sqrt(1-Alpha_bar[t])
                    #second_option = second_option - epsilon_theta2
                    x = old_x.clone().detach() + cicle_x.clone().detach()
                elif k == 0:
                    x = (x - ((1 - Alpha[t])/(1-Alpha_bar[t]))*epsilon_theta ) 
                
                

                

                #wandb.log({'epoch': t+k*T})

                if (t > 0) and (k > -1):

                    #x = x + (torch.sqrt((dk[:,:,k].unsqueeze(-1)))*(1-Alpha[t]) + torch.sqrt((dk[:,:,:k].sum(dim=-1)).unsqueeze(-1))) * std_normal(size)  
                    #x = x + torch.sqrt(dk[:,:,k].unsqueeze(-1)*Sigma[t] + (1 - Alpha[t])/(1-Alpha_bar[t])) * std_normal(size)
                    x = x + (torch.sqrt((dk[:,:,k].unsqueeze(-1))*((Sigma[t])))) * std_normal(size)  + torch.sqrt(dk[:,:,:k].sum(dim=-1)).unsqueeze(-1)*std_normal(size) 
                elif (t == 0) and (k > (max_components-max_components_gen)+1):
                    if k-1>0:
                        #cicle_x =  torch.sqrt(((dk[:,:,:k].sum(dim=-1)).unsqueeze(-1))) * std_normal(size)
                        #cicle_x =  torch.sqrt(((dk[:,:,k-1]).unsqueeze(-1))) * std_normal(size)
                        #old_x = old_x + cicle_x.clone().detach()
                        old_x = x.clone().detach()
                        cicle_x = torch.zeros_like(x)

                        #old_x = old_x + cicle_x.clone().detach()
                        #x = old_x+cicle_x
                    
                    else:
                        
                        #old_x = old_x + cicle_x.clone().detach()
                        old_x = x.clone().detach()
                        cicle_x =  torch.zeros_like(x)

                        #x = x+cicle_x
                        #print('the_end_of_the_diffusion_process')
                    #plt.plot(old_x[0,:,700:].T.cpu().detach().numpy())
                    #plt.show()
                    #x = x + ((dk[:,:,:-(max_components-k)-1].sum(dim=-1)).unsqueeze(-1)) * std_normal(size) # add the variance term to x_{t-1}

                    
                #x_step = x.clone().detach().cpu().numpy()    
                #epsilon_theta_save = epsilon_theta.clone().detach().cpu().numpy()
                #second_option_save = second_option.clone().detach().cpu().numpy()
                #x_step = x.clone().detach().cpu().numpy()
                steps.append(x_step)
                #epsilon_theta_list.append(epsilon_theta_save)
                #steps2.append(cicle_x.clone().detach().cpu().numpy())
    
        generated_audio = x
    return generated_audio, steps #, epsilon_theta_list, steps2

    
def frequency_finder(y, n_frequencies=None):
    # baseline noise in fft domain
    batch_size, channels, length = y.shape
    
    omega = torch.fft.rfft(y)
    freqs = torch.fft.rfftfreq(length, 1)  # Get frequency axis from the time axis
    mags = torch.abs(omega)
    
    
    diff_1 = torch.diff(torch.sign(torch.diff(mags, dim=2)),dim=2)
    inflection2 = torch.sign(diff_1)
    peaks_mask =  (inflection2[:, :, 1:] < 0)
    adjusted_peaks_mask = torch.nn.functional.pad(peaks_mask, (2, 1), "constant", 0)
    peak_magnitudes = mags * adjusted_peaks_mask

    if n_frequencies is None:
        b_noise = torch.mean(mags[:,:,1:],axis=(2)) + torch.sqrt(torch.var(mags[:,:,1:],axis=(2)))
        b_noise = b_noise.unsqueeze(2)
        n_frequencies = torch.sum(peak_magnitudes[:,:,1:] > b_noise,axis=2) #TODO change frequency finder to identify white noise level ( tipo quando a derivada da mag fica 0.)
    else:
        n_frequencies = torch.tensor(n_frequencies).expand(batch_size, channels)
        
    h_p_m = torch.zeros(y.shape[0], y.shape[1], n_frequencies.max(), device=y.device)
    h_p_f = torch.zeros(y.shape[0], y.shape[1], n_frequencies.max(), device=y.device)
    h_p_c = torch.zeros(y.shape[0], y.shape[1], n_frequencies.max(), device=y.device, dtype=torch.complex64)
    h_p_i = torch.zeros(y.shape[0], y.shape[1], n_frequencies.max(), device=y.device, dtype=torch.long)

    for batch_idx in range(y.shape[0]):
        for channel_idx in range(y.shape[1]):
            num_freqs = n_frequencies[batch_idx, channel_idx]

            if num_freqs > 0:
                # Get the indices of the significant magnitudes
                top_mags, top_indices = peak_magnitudes[batch_idx,channel_idx].topk(num_freqs)
                
                
                # Store the top magnitudes
                h_p_m[batch_idx, channel_idx, :num_freqs] = top_mags #magnitudes 
                # Store the corresponding frequencies
                h_p_f[batch_idx, channel_idx, :num_freqs] = freqs.cuda()[top_indices] # frequencies
                # store the corresponding complex values
                h_p_c[batch_idx, channel_idx, :num_freqs] = omega[batch_idx,channel_idx][top_indices]
                # store the corresponding indices
                h_p_i[batch_idx, channel_idx, :num_freqs] = top_indices

    
    return h_p_f, h_p_m, h_p_i, h_p_c

def fixed_frequency_finder(y, n_frequencies):

    batch_size, channels, length = y.shape

    omega = torch.fft.rfft(y)
    freqs = torch.fft.rfftfreq(length, 1)  # Get frequency axis from the time axis
    mags = torch.abs(omega)

    top_k_values_mags, top_k_indices = mags[:,:,1:].topk(n_frequencies) #don't count DC

    top_k_indices = top_k_indices + 1

    h_p_m = top_k_values_mags 

    h_p_f = freqs.cuda()[top_k_indices]

    h_p_i = top_k_indices

    h_p_c = torch.gather(omega, dim=-1, index=top_k_indices)

    h_p_i = h_p_i.to(torch.int64)



    return h_p_f, h_p_m, h_p_i, h_p_c


    

def sort_components(components,components_var=None):

    '''
    Sort components based on variance 
    the smaller components are the ones diffused first

    out:
    '''
    if components_var is not None:
        _, sorted_indices = torch.sort(components_var, descending=False, dim=2)
    else:
        _, sorted_indices = torch.sort(components.var(dim=-1), descending=False, dim=2)
    #print(sorted_indices)
    # Adjust sorted_indices to work with components
    # We need to expand sorted_indices to the shape (50, 14, 6, 100) to match components
    sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, components.size(3))

    # Reorder components based on sorted indices
    sorted_components = torch.gather(components, dim=2, index=sorted_indices_expanded)

    return sorted_components, sorted_indices

# Reorder components based on sorted indices
# Use `torch.gather` to apply the sorted indices to reorder components
#    sorted_components = torch.gather(components, dim=-1, index=sorted_indices.unsqueeze(2).expand(-1, -1, components.size(2), -1))


def synthetic_fft(signal,n_frequencies=None,fixed = True,bands=None):
    """
    Perform iFFT on the input signal and return the output and the components of the FFT.

    return output, inverse_fft_components
    """
    #
    batch_size, channels, signal_length = signal.shape
    
    if n_frequencies is None:
        if fixed == False:
            signal_freq, magnitude,h_p_i, h_p_c = frequency_finder(signal)  # signal freq is 
        else:
            signal_freq, magnitude,h_p_i, h_p_c = fixed_frequency_finder(signal,n_frequencies)
    else:
        if fixed == False:
            signal_freq, magnitude,h_p_i, h_p_c = frequency_finder(signal)
        else:
            signal_freq, magnitude,h_p_i, h_p_c = fixed_frequency_finder(signal,n_frequencies)
    
    magnitude = (magnitude / signal_length) * 2

    #l_signals = torch.empty(0)
    #for i in range(len(signal_freq)):
    #    y = magnitude[:, i].unsqueeze(1) * torch.sin(torch.arange(0, signal_length) * 2 * np.pi * signal_freq[:, i].unsqueeze(1))
    #    l_signals.append(y)
        
    #l_signals_tensor = torch.stack(l_signals)

    fft_length = signal_length // 2 + 1

    _,_,top_k = h_p_i.shape

    # Expand dimensions for h_p_i and h_p_c
    h_p_i_expanded = h_p_i.unsqueeze(-1)  # Shape: (batch_size, channels, 4, 1)
    h_p_c_expanded = h_p_c.unsqueeze(-1)  # Shape: (batch_size, channels, 4, 1)


    # Create a new tensor with an additional dimension
    synth = torch.zeros(
        batch_size, channels, top_k, fft_length, 
        dtype=torch.complex64, 
        device='cuda'
    )

    # Scatter the values in h_p_c along the new dimension
    synth_complete = synth.scatter_(-1, h_p_i_expanded, h_p_c_expanded)

    # Perform inverse FFT on the synthetic components

    synth_y = torch.fft.irfft(synth_complete, dim=-1)
    
    #synth_y = (magnitude[:,:]).unsqueeze(3) * torch.sin(torch.arange(0, signal_length).cuda() * 2 * torch.pi * signal_freq[:, :].unsqueeze(3))
    output = torch.sum(synth_y, dim=2)
    
    
    remainder_component = signal - output
    remainder_component = remainder_component.unsqueeze(2)  # Make it a "new" component by adding the third dimension
    synth_y = torch.cat((synth_y, remainder_component), dim=2)
    return output, synth_y

def nanvar(tensor, dim, unbiased=True):
    """Computes variance ignoring NaNs (similar to torch.var but NaN-safe)."""
    mask = ~torch.isnan(tensor)  # Get valid (non-NaN) values
    count = mask.sum(dim, keepdim=True)  # Number of valid elements per dim
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)  # Compute NaN-safe mean
    
    # Compute variance manually: sum((x - mean)^2) / count
    var = torch.where(mask, (tensor - mean) ** 2, torch.tensor(0.0, device=tensor.device))
    var = var.sum(dim, keepdim=True) / (count - int(unbiased))  # Apply Bessel's correction if unbiased
    
    return var.squeeze(dim)  # Remove the extra dimension

def sort_components_(components, components_var=None):
    '''
    Sort components based on variance 
    the smaller components are the ones diffused first,
    but the residual (last component) is always moved to the first position.
    '''
    if components_var is not None:
        _, sorted_indices = torch.sort(components_var[:, :, :-1], descending=False, dim=2)
    else:
        _, sorted_indices = torch.sort(components[:, :, :-1].var(dim=-1), descending=False, dim=2)

    # Move residual to first position
    residual = components[:, :, -1:, :]  # Shape: (50, 14, 1, 100)

    
    
    # Reorder components based on sorted indices (excluding the residual)

    sorted_indices_excluding_last = sorted_indices[:, :, :]  # Shape: (50, 14, 5)
    sorted_indices_expanded = sorted_indices_excluding_last.unsqueeze(-1).expand(components.size(0), -1, -1, components.size(3))

    sorted_components = torch.gather(components, dim=2, index=sorted_indices_expanded)


    # Concatenate residual as the first component
    new_components = torch.cat([residual, sorted_components], dim=2)

    return new_components, sorted_indices


def log_noise_level(diffusion_steps, epsilon_theta,commit=False):
    """
    Logs the noise level across different diffusion steps.
    
    Args:
        diffusion_steps (Tensor): The diffusion step indices (shape: [B,1,1]).
        epsilon_theta (Tensor): The predicted noise (shape: [B, C, T]).
    """
    stepwise_noise = {}  # Dictionary to hold step-wise noise levels

    for step in diffusion_steps.unique():
        mask = diffusion_steps.squeeze() == step  # Find batch elements at this step
        noise_at_step = epsilon_theta[mask]  # Select corresponding noise values
        
        if noise_at_step.numel() > 0:  # Avoid empty selection
            noise_norm = nanvar(noise_at_step, dim=-1).mean().item()  # Compute mean L2 norm
            stepwise_noise[f"noise_level_step_{step.item()}"] = noise_norm  # Save noise level
    
    #wandb.log(stepwise_noise,commit=commit)
    return stepwise_noise

def log_loss_level(diffusion_steps, epsilon_theta,commit=False):
    """
    Logs the noise level across different diffusion steps.
    
    Args:
        diffusion_steps (Tensor): The diffusion step indices (shape: [B,1,1]).
        epsilon_theta (Tensor): The predicted noise (shape: [B, C, T]).
    """
    stepwise_noise = {}  # Dictionary to hold step-wise noise levels

    for step in diffusion_steps.unique():
        mask = diffusion_steps.squeeze() == step  # Find batch elements at this step
        noise_at_step = epsilon_theta[mask]  # Select corresponding noise values
        
        if noise_at_step.numel() > 0:  # Avoid empty selection
            noise_norm = torch.nanmean(noise_at_step, dim=-1).mean().item()  # Compute mean L2 norm
            stepwise_noise[f"noise_loss_step_{step.item()}"] = noise_norm  # Save noise level
    
    #wandb.log(stepwise_noise,commit=commit)
    return stepwise_noise



def log_noise_bar_chart(stepwise_noise, log_time):
    """
    Logs a bar chart for noise levels at different diffusion steps.

    Args:
        stepwise_noise (dict): Noise levels per step (e.g., {"noise_level_step_1": val, ...}).
        log_time (int): Current logging step.
    """
    # Extract step numbers and corresponding values
    steps = [int(k.split("_")[-1]) for k in stepwise_noise.keys()]
    values = list(stepwise_noise.values())
    unique_steps = len(set(steps))
    # Create a W&B Table
    #table2 = wandb.Table(columns=["Diffusion Step", "Noise Level"])

    #print(steps, values)
    #for step, value in zip(steps, values):
    #    table2.add_data(step, value)
    fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

    ax.plot(steps, values)
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Noise Var")

    ax.text(0.05, 0.95, f"# unique_steps: {unique_steps}",  
        transform=ax.transAxes, fontsize=10, ha="left", va="top",
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
    #wandb.log({f"noise_bar_chart_time_{log_time}": wandb.plot.bar(table2, "Diffusion Step", "Noise Level", title="Noise Levels per Diffusion Step")},commit=False)
    wandb.log({f"noise_line_plot": wandb.Image(plt)},commit=False)

    plt.close(fig)

    
def log_loss_bar_chart(stepwise_noise, log_time):
    """
    Logs a bar chart for noise levels at different diffusion steps.

    Args:
        stepwise_noise (dict): Noise levels per step (e.g., {"noise_level_step_1": val, ...}).
        log_time (int): Current logging step.
    """
    # Extract step numbers and corresponding values
    steps = [int(k.split("_")[-1]) for k in stepwise_noise.keys()]
    values = list(stepwise_noise.values())

    # Create a W&B Table
    #table = wandb.Table(columns=["Diffusion Step", "Loss"])

    #print(steps, values)
    #for step, value in zip(steps, values):
    #    table.add_data(step, value)
    fig, ax = plt.subplots(dpi=300, figsize=(10, 6))

    ax.plot(steps, values)
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Loss")
    #wandb.log({f"loss_bar_chart_time_{log_time}": wandb.plot.bar(table, "Diffusion Step", "Loss", title="loss per Diffusion Step")},commit=False)
    wandb.log({f"loss_line_plot": wandb.Image(plt)},commit=False)

    plt.close(fig)

def calc_quantile_CRPS(target, torch_generated_audio_samples,loss_mask,quantiles=None):

    """
    Computes quantile-based CRPS from sample-based forecasts.
    # note: the median for even number of samples is the lowest value, while quantile(0.5) is the average of the two middle values.
    # use quantile(0.5) for the median and when possible an odd number of samples 
    Args:
        target: [B,C,T]
        forecast: [N,B, 1, T]
        eval_points: [B, C,T] binary mask of points to evaluate

    Returns:
        scalar CRPS (float)
    """
    if quantiles is None:
        quantiles = torch.arange(0.05, 0.95, 0.05, device='cpu')
    
    target = target.cpu()
    loss_mask =  loss_mask.cpu()

    quantiled_samples = torch.quantile(torch_generated_audio_samples, quantiles, dim=0)
    CRPS = 0
    errors_all = target - quantiled_samples  # [Q,B, C, T]

    for i in range(len(quantiles)):

        q = quantiles[i].unsqueeze(0).unsqueeze(0)
        pinball = 2*torch.maximum(q * errors_all[i], (q - 1) * errors_all[i])
        CRPS += (pinball * loss_mask).mean()  

    return (CRPS / len(quantiles)).item()

def calc_sample_CRPS(target, torch_generated_audio_samples, loss_mask):
    """
    Computes sample-based CRPS using empirical forecast distribution.

    Args:
        target: [B, C, T]
        torch_generated_audio_samples: [N, B, C, T]  # N samples
        loss_mask: [B, C, T]  # binary mask of evaluation points

    Returns:
        scalar CRPS (float)
    """
    # Ensure same device and dtype
    device = torch_generated_audio_samples.device
    target = target.to(device)
    loss_mask = loss_mask.to(device)

    samples = torch_generated_audio_samples  # [N, B, C, T]
    N = samples.shape[0]

    # Term 1: E|Y - x|
    abs_diff_target = torch.abs(samples - target.unsqueeze(0))  # [N, B, C, T]
    term1 = abs_diff_target.mean(dim=0)  # [B, C, T]

    # Term 2: 0.5 * E|Y - Y'|
    samples1 = samples.unsqueeze(0)  # [1, N, B, C, T]
    samples2 = samples.unsqueeze(1)  # [N, 1, B, C, T]
    abs_diff_samples = torch.abs(samples1 - samples2)  # [N, N, B, C, T]
    term2 = 0.5 * abs_diff_samples.mean(dim=(0, 1))  # [B, C, T]

    crps = (term1 - term2) * loss_mask  # Apply mask
    return crps.mean().item()



def training_loss_new(net, loss_fn, X, diffusion_hyperparams,max_components, only_generate_missing=1,monte_carlo=True,logging=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]


    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
     # randomly sample diffusion steps from 1~T (choose a random step for each batch element)

    z = std_normal(audio.shape)

    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()
    components_steps = torch.randint(max_components+1, size=(B, 1, 1)).cuda()

    if only_generate_missing == 0:
        #z = audio * mask.float() + z * (1 - mask).float()
    #transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
    #    1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
        transformed_X,Transformed_X_prior, transformed_X_noise , _dh = apply_forward(audio,_dh, diffusion_steps, components_steps,max_components = max_components, monte_carlo=monte_carlo)
    # I want the output to be transformed_X 
    elif only_generate_missing == 1:
    
        transformed_X,transformed_X_prior, transformed_X_noise , _dh, dk_scaled = apply_forward(audio,_dh, diffusion_steps, components_steps,max_components = max_components, monte_carlo=monte_carlo)
        transformed_X = audio * mask.float() +  transformed_X * (1 - mask).float()
        transformed_X_prior = audio * mask.float() +  transformed_X_prior * (1 - mask).float()
        transformed_X_noise = audio * mask.float() + transformed_X_noise * (1-mask).float()
    diffusion_steps = _dh['t']
    components_steps = _dh['k']
    epsilon_theta = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1)+T*components_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta
    
    if logging:
        if only_generate_missing == 1:
            stepwise_noise = log_noise_level(diffusion_steps.view(B, 1)+T*components_steps.view(B, 1),torch.where(loss_mask, epsilon_theta, torch.nan)) # this applies the mask on epsilon theta. because we wnat to 
            stepwise_loss = log_loss_level(diffusion_steps.view(B, 1)+T*components_steps.view(B, 1),(torch.where(loss_mask, epsilon_theta, torch.nan)-torch.where(loss_mask, transformed_X_noise, torch.nan))**2)
            log_noise_bar_chart(stepwise_noise,logging)
            log_loss_bar_chart(stepwise_loss,logging)
        elif only_generate_missing == 0:
            stepwise_noise = log_noise_level(diffusion_steps.view(B, 1)+T*components_steps.view(B, 1), epsilon_theta)
            stepwise_loss = log_loss_level(diffusion_steps=(diffusion_steps.view(B, 1)+T*components_steps.view(B, 1)), epsilon_theta = (epsilon_theta-transformed_X_noise)**2)
            log_noise_bar_chart(stepwise_noise,logging)
            log_loss_bar_chart(stepwise_loss,logging)
       
    if isinstance(loss_fn, torch.nn.MSELoss):
        if loss_mask is not None:
            return loss_fn(epsilon_theta[loss_mask], transformed_X_noise[loss_mask])
        else:
            return loss_fn(epsilon_theta, transformed_X_noise)

    if only_generate_missing == 1:
        if isinstance(loss_fn, torch.nn.MSELoss):
            return loss_fn(epsilon_theta[loss_mask], transformed_X_noise[loss_mask])
        else:
            return weighted_mse_loss(epsilon_theta, transformed_X_noise,loss_mask = loss_mask, expanded_dk_scaled = dk_scaled)
    elif only_generate_missing == 0:
        if isinstance(loss_fn, torch.nn.MSELoss):
            return loss_fn(epsilon_theta, transformed_X_noise)
        else:
            return weighted_mse_loss(epsilon_theta, transformed_X_noise,loss_mask = None, expanded_dk_scaled = dk_scaled)


def weighted_mse_loss(epsilon_theta, transformed_X_noise, loss_mask=None, expanded_dk_scaled = 1):
    """
    Computes a weighted mean squared error loss.

    Parameters:
    - epsilon_theta: Predicted noise tensor
    - transformed_X_noise: Target noise tensor
    - loss_mask: Boolean mask to select valid entries
    - expanded_dk_scaled: Weight scaling factor tensor

    Returns:
    - Scalar tensor representing the weighted MSE loss
    """
    dk_scaled_expanded = expanded_dk_scaled.repeat(1,1,loss_mask.shape[-1])[loss_mask] #yes, i know the name, but it gets expanded inside the function

    if loss_mask is None:
        loss_mask = torch.ones_like(epsilon_theta, dtype = torch.bool)
    squared_errors = torch.pow((epsilon_theta[loss_mask] - transformed_X_noise[loss_mask]) /(dk_scaled_expanded), 2)
    # squared_errors = torch.pow((epsilon_theta[loss_mask] - transformed_X_noise[loss_mask]) * (1- expanded_dk_scaled), 2)
    weighted_squared_errors = squared_errors 
    return weighted_squared_errors.mean()
    

def get_optimizer(net, learning_rate, scheduler_type="cosine", T_max=100, eta_min=1e-6):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        scheduler = None

    return optimizer, scheduler

def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))  # lenght of series indexes
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        idx = perm[0:k]
        mask[:, channel][idx] = 0

    return mask


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask

def get_mask_forecast(sample, k):
    """Get mask of same segments (forecast missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = list_of_segments_index[1]
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask

def get_mask_forecast(sample, k):
    """Get mask of same segments (forecast missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = list_of_segments_index[1]
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask

def get_mask(batch, masking,k):
    "wrapper function to get mask based on masking type"
    if masking == 'rm':
        transposed_mask = get_mask_rm(batch[0], k)
        #TODO random masking is wrong, needs to be the same for all features
    elif masking == 'mnr':
        transposed_mask = get_mask_mnr(batch[0], k)
    elif masking == 'bm':
        transposed_mask = get_mask_bm(batch[0], k)
    elif masking == 'forecast':
        transposed_mask = get_mask_forecast(batch[0],k)

    mask = transposed_mask.permute(1, 0)
    mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
    loss_mask = ~mask.bool()

    return mask, loss_mask


