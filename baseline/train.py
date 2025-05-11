import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.util_old import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from utils.util_old import get_mask_mnr, get_mask_bm, get_mask_rm, get_mask_forecast

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer
from imputers.CSDIImputer import CSDIImputer


from json import JSONEncoder
import json

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*keyword argument dtype in Genred is deprecated.*")



def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k,
          fixed_components):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    base_directory = "/data/f.caldas/diffusion"
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    final_directory = os.path.join(base_directory, output_directory)
    if not os.path.isdir(final_directory):
        os.makedirs(final_directory)
        os.chmod(final_directory, 0o775)

    print("Final directory created:", final_directory, flush=True)



    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    if fixed_components == 0:
        diffusion_hyperparams['fixed'] = False
    else:
        diffusion_hyperparams['fixed'] = True

    # predefine model
    if use_model == 0:
        net = DiffWaveImputer(**model_config).cuda()
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).cuda()
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).cuda()
    elif use_model == 3:
        net = CSDIImputer(**model_config).cuda()
    else:
        print('Model chosen not available.')
    print_size(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    while ckpt_iter > 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
            break
            
        except:
            if ckpt_iter > 0:
                ckpt_iter -= iters_per_ckpt
                print('Failed to load model at iteration {}, trying previous iteration'.format(ckpt_iter))
            else:
                ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
        
        
        
        
    
    ### Custom data loading and reshaping ###
        
    if trainset_config['train_data_path'] is not None:    
        if trainset_config['train_data_path'] == "./datasets/Mujoco/train_mujoco.npy":
            dat_path = "/data/f.caldas/csdi/datasets/Mujoco/train_mujoco.npy"
            if os.path.exists(dat_path):
                print('importing from data server')
                training_data = np.load(dat_path)
            else:
                training_data = np.load(trainset_config['train_data_path'])
            training_data = np.split(training_data, 160, 0)
            training_data = np.array(training_data)
            training_data = torch.from_numpy(training_data).float().cuda()
            print('Data loaded - Mujoco')
        elif trainset_config['train_data_path'] == "./datasets/Electricity/train_electricity.npy":
            #dat_path = "/data/f.caldas/csdi/datasets/Electricity/train_electricity.npy"
            #if os.path.exists(dat_path):
            #    print('importing from data server')
            #    training_data = np.load(dat_path)
            #else:
            #    training_data = np.load(trainset_config['train_data_path'])
            #option1_data = training_data.reshape(-1, 100,37)
            #option2_data = option1_data.transpose(0,2,1).reshape(-1,1, 100)
            #option2_data = option2_data.transpose(0,2,1)
            #option2_data = np.split(option2_data, 370, 0)
            #option2_data = np.array(option2_data)
            #training_data = torch.from_numpy(option2_data).float().cuda()
            #print('Data loaded - Electricity')

            class ElectricityTrainingDataset(Dataset):
                def __init__(self, trainset_config):
                    dat_path = "/data/f.caldas/csdi/datasets/Electricity/train_electricity.npy"
                    if os.path.exists(dat_path):
                        print('Importing from data server...')
                        training_data = np.load(dat_path)
                    else:
                        training_data = np.load(trainset_config['train_data_path'])
                    option1_data = training_data.reshape(-1, 100,37)
                    option2_data = option1_data.transpose(0,2,1).reshape(-1,1, 100)
                    option2_data = option2_data.transpose(0,2,1)
                    #option2_data = np.split(option2_data, 370, 0)
                    option2_data = np.array(option2_data)
                    self.training_data = torch.from_numpy(option2_data).float()

                def __len__(self):
                    return self.training_data.size(0)

                def __getitem__(self, idx):
                    return self.training_data[idx]
            
            dataset = ElectricityTrainingDataset(trainset_config)
            print('Data loaded - ettm1',flush=True)
            if use_model !=0:
                training_data = DataLoader(dataset, batch_size=132, shuffle=False)
            else:
                training_data = DataLoader(dataset, batch_size=264, shuffle=False)

        elif trainset_config['train_data_path'] == "./datasets/ETTm1/train_ettm1_1056.npy":
            #ettm1_path = "/data/f.caldas/csdi/datasets/ETTm1/train_ettm1_1056.npy"
            #if os.path.exists(ettm1_path):
            #    print('importing from data server')
           #     training_data = np.load(ettm1_path)
            #else:
            #    training_data = np.load(trainset_config['train_data_path'])
            #training_data = np.split(training_data, 571, 0)
            #training_data = np.array(training_data)
            #training_data = torch.from_numpy(training_data).float().cuda()
            #print('Data Loaded -ettm1')

            class ETTm1TrainingDataset(Dataset):
                def __init__(self, trainset_config):
                    ettm1_path = "/data/f.caldas/csdi/datasets/ETTm1/train_ettm1_1056.npy"
                    if os.path.exists(ettm1_path):
                        print('Importing from data server...')
                        training_data = np.load(ettm1_path)
                    else:
                        training_data = np.load(trainset_config['train_data_path'])
                    #training_data = np.split(training_data, 571, 0)
                    training_data = np.array(training_data)
                    self.training_data = torch.from_numpy(training_data).float()

                def __len__(self):
                    return self.training_data.size(0)

                def __getitem__(self, idx):
                    return self.training_data[idx]
            
            dataset = ETTm1TrainingDataset(trainset_config)
            print('Data loaded - ettm1',flush=True)
            if use_model !=0:
                training_data = DataLoader(dataset, batch_size=8, shuffle=True)
            else:
                training_data = DataLoader(dataset, batch_size=32, shuffle=True)
            
        elif trainset_config['train_data_path'] == "./datasets/PTB-XL/train_ptbxl_1000.npy":
            #dat_path = "/data/f.caldas/csdi/datasets/PTB-XL-1000/train_ptbxl_1000.npy"
            #if os.path.exists(dat_path):
            #    print('importing from data server')
            #    training_data = np.load(dat_path)
            #else:
            #    training_data = np.load(trainset_config['train_data_path'])
            #training_data = training_data.transpose(0,2,1)
            #training_data = np.split(training_data, 163, 0)
            #training_data = np.array(training_data)
            #training_data = torch.from_numpy(training_data).float().cuda()
            #print('Data loaded - PTB-Xl')

            class PTBXLTrainingDataset(Dataset):
                def __init__(self, trainset_config):
                    dat_path = "/data/f.caldas/csdi/datasets/PTB-XL-1000/train_ptbxl_1000.npy"
                    if os.path.exists(dat_path):
                        print('Importing from data server...')
                        training_data = np.load(dat_path)
                    else:
                        training_data = np.load(trainset_config['train_data_path'])
                    # Transpose to match the desired shape
                    training_data = training_data.transpose(0,2,1)
                    training_data = np.array(training_data)
                    self.training_data = torch.from_numpy(training_data).float()

                def __len__(self):
                    return self.training_data.size(0)

                def __getitem__(self, idx):
                    return self.training_data[idx]
            # Instantiate Dataset and DataLoader
            dataset = PTBXLTrainingDataset(trainset_config)
            print('Data loaded - PTB-Xl',flush=True)
            if use_model !=0:
                training_data = DataLoader(dataset, batch_size=8, shuffle=False)
            else:
                training_data = DataLoader(dataset, batch_size=64, shuffle=False)

    

    
    
    
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch in training_data:
            if not batch.is_cuda:
                batch = batch.to('cuda')
            if masking == 'rm':
                transposed_mask = get_mask_rm(batch[0], missing_k)
                #TODO random masking is wrong, needs to be the same for all features
            elif masking == 'mnr':
                transposed_mask = get_mask_mnr(batch[0], missing_k)
            elif masking == 'bm':
                transposed_mask = get_mask_bm(batch[0], missing_k)
            elif masking == 'forecast':
                transposed_mask = get_mask_forecast(batch[0],missing_k)

            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            batch = batch.permute(0, 2, 1)

            assert batch.size() == mask.size() == loss_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),'loss':loss.item()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)
            
            
            class EncodeTensor(JSONEncoder,Dataset):
                def default(self, obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu().detach().numpy().tolist()
                    return super(EncodeTensor, self).default(obj)



            if n_iter > 0 and n_iter % (iters_per_ckpt*2) == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                with open(os.path.join(final_directory, checkpoint_name), 'w') as json_file:
                    json.dump(net.state_dict(), json_file,cls=EncodeTensor)
                print('model at iteration %s is json saved' % n_iter)

            n_iter += 1
    
    print('Training finished')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSDS4.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    train(**train_config)
