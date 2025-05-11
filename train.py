import os
import argparse
import json
import numpy as np
import torch
import time
import torch.nn as nn

import wandb


from utils.util import find_max_epoch, print_size, training_loss_new, calc_diffusion_hyperparams,calc_diffusion_hyperparams2,masked_components_fft_amplitude
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, get_mask_forecast, get_mask , weighted_mse_loss,get_optimizer,sampling_new2

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer
from imputers.CSDIImputer import CSDIImputer

from dtl_utils.datasets_torchloaders import MujocoTrainingDataset, MujocoTestDataset,BaseDatasetClass
from dtl_utils.datasets_torchloaders import ETTm1TestDataset, ElectricityTestDataset, PTBXLTestDataset, ETTm1TrainingDataset, ElectricityTrainingDataset, PTBXLTrainingDataset,Synth1TrainingDataset, Synth1ValidationDataset
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error



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
          max_components,
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
    local_path = "T{}_beta0{}_betaT{}_n{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"],
                                              max_components)
    
    loss_best =1000

    # Get shared output_directory ready
    #base_directory = "/data/f.caldas/diffusion"
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    #final_directory = os.path.join(base_directory, output_directory)
    #if not os.path.isdir(final_directory):
    #    os.makedirs(final_directory)
    #    os.chmod(final_directory, 0o775)

    #print("Final directory created:", final_directory, flush=True)

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
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer, scheduler = get_optimizer(net, learning_rate=learning_rate, T_max=100,scheduler_type='nao quero')
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

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
            
            if 'loss' in checkpoint:
                loss_best = checkpoint['loss']

            print('Successfully loaded model at iteration {}'.format(ckpt_iter),flush = True)
            break

        except:
            if ckpt_iter > 0:
                ckpt_iter -= iters_per_ckpt
                print('Failed to load model at iteration {}, trying previous iteration'.format(ckpt_iter),flush = True)
            else:
                ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.',flush=True)
        
        
    
    ### Custom data loading and reshaping ###
        
        

    if trainset_config['train_data_path'] is not None:    
        if trainset_config['train_data_path'] == "./datasets/Mujoco/train_mujoco.npy":

            dataset = MujocoTrainingDataset(trainset_config)

            do_validation = 'val_data_path' in trainset_config

            if 'batch_size' in trainset_config:
                batch_size = trainset_config['batch_size']
            else:
                batch_size = 128   

            if do_validation:
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            else:
                train_dataset = dataset
                val_dataset = dataset 
            
            print('Data loaded - mujoco',flush=True)
            if use_model !=0:
                training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(val_dataset, batch_size=500, shuffle=False)
            else:
                training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(val_dataset, batch_size=1000, shuffle=False)

            
            #dat_path = "/data/f.caldas/csdi/datasets/Mujoco/train_mujoco.npy"
            #if os.path.exists(dat_path):
            #    print('importing from data server')
            #    training_data = np.load(dat_path)
            #else:
            #    training_data = np.load(trainset_config['train_data_path'])
            #training_data = np.split(training_data, 160, 0)
            #training_data = np.array(training_data)
            #training_data = torch.from_numpy(training_data).float().cuda()
            #print('Data loaded - Mujoco')

            #do_validation = True
            #sval_dataset = training_data[0:20]


        elif trainset_config['train_data_path'] == "./datasets/Electricity/train_electricity.npy":

            dataset = ElectricityTrainingDataset(trainset_config)

            do_validation = 'val_data_path' in trainset_config

            if 'batch_size' in trainset_config:
                batch_size = trainset_config['batch_size']
            else:
                batch_size = 128

            if do_validation:
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            else:
                train_dataset = dataset
                val_dataset = dataset

            print('Data loaded - electricity',flush=True)
            if use_model !=0:
                training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(val_dataset, batch_size=500, shuffle=False)
            else:
                training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(val_dataset, batch_size=1000, shuffle=False)

        elif trainset_config['train_data_path'] == "./datasets/ETTm1/train_ettm1_1056.npy":


            
            dataset = ETTm1TrainingDataset(trainset_config)

            do_validation = 'val_data_path' in trainset_config

            if 'batch_size' in trainset_config:
                batch_size = trainset_config['batch_size']
            else:
                batch_size = 128

            if do_validation:
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            print('Data loaded - ettm1',flush=True)
            if use_model !=0:
                training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(val_dataset, batch_size=100, shuffle=False)
            else:
                training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(val_dataset, batch_size=100, shuffle=False)
            
        elif trainset_config['train_data_path'] == "./datasets/PTB-XL/train_ptbxl_1000.npy":
            # PTB-XL
            # Instantiate Dataset and DataLoader
            dataset = PTBXLTrainingDataset(trainset_config)

            do_validation = 'val_data_path' in trainset_config

            if 'batch_size' in trainset_config:
                batch_size = trainset_config['batch_size']
            else:
                batch_size = 128

            if do_validation:
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            else:
                train_dataset = dataset
                val_dataset = dataset
            
            print('Data loaded - PTB-Xl',flush=True)
            if use_model !=0:
                training_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(val_dataset, batch_size=500, shuffle=False)

            else:
                training_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(val_dataset, batch_size=100, shuffle=False)
        elif trainset_config['train_data_path'] == "./datasets/synth1/y_train.npy":
            dataset = Synth1TrainingDataset(trainset_config)  # Define dataset separately
            if 'batch_size' in trainset_config:
                batch_size = trainset_config['batch_size']
            else:
                batch_size = 128

            do_validation = 'val_data_path' in trainset_config

           

            if do_validation:
                dataset_validation = Synth1ValidationDataset(trainset_config)
            else:
                dataset_validation = dataset
                

            if use_model !=0:
                training_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(dataset_validation, batch_size=100, shuffle=False)

            else:
                training_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(dataset_validation, batch_size=200, shuffle=False)
        
        else:
                dataset = BaseDatasetClass(trainset_config)

                # Determine if validation should be done
                do_validation = 'val_data_path' in trainset_config

                # Use batch size from config or fallback to default
                batch_size = trainset_config.get('batch_size', batch_size)

                if do_validation:
                    train_size = int(0.8 * len(dataset))
                    val_size = len(dataset) - train_size
                    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                else:
                    train_dataset = dataset
                    val_dataset = dataset

                print(f"Data loaded - {dataset.name}", flush=True)

                # Define different val batch sizes depending on model type
                val_batch_size = 500

                training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                validation_data = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)


    if trainset_config['loss']== "mse":
        mse_loss = torch.nn.MSELoss()
    elif trainset_config['loss']== "wmse":
        mse_loss = weighted_mse_loss
    else:
        print("Loss not available")
        raise ValueError("Loss not available")
    
    # training
    n_iter = ckpt_iter + 1
    epoch = 0
    while n_iter < n_iters + 1:
        epoch_start_time = time.time()
        for batch in training_data:
            
            batch = batch.cuda()
            mask,loss_mask = get_mask(batch, masking, missing_k)  # Define mask generation separately


            batch = batch.permute(0, 2, 1)

            assert batch.size() == mask.size() == loss_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            #net, loss_fn, X, diffusion_hyperparams,max_components, only_generate_missing=1

            
            loss = training_loss_new(net, mse_loss, X, diffusion_hyperparams,max_components=max_components,
                                 only_generate_missing=only_generate_missing,monte_carlo=False,logging=None)

            loss.backward()
            optimizer.step()


            if n_iter > 0 and n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

                if do_validation:


                    net.eval() 
                    val_losses = 0 # Set model to evaluation mode
                    with torch.no_grad():
                        for val_batch in validation_data:  # Assume validation_data is a DataLoader
                            val_batch = val_batch.cuda()
                            val_mask, val_loss_mask = get_mask(val_batch, masking, missing_k)
                            
                            val_batch = val_batch.permute(0, 2, 1)

                            X_val = val_batch, val_batch, val_mask, val_loss_mask

                            #X = batch, batch, mask, loss_mask
                            #net, loss_fn, X, diffusion_hyperparams,max_components, only_generate_missing=1
                            loss_val = training_loss_new(net, torch.nn.MSELoss(), X_val, diffusion_hyperparams,max_components=max_components,
                                                only_generate_missing=only_generate_missing,monte_carlo=False,logging=n_iter)
                            

                            
                            
                            val_losses = loss_val.item() + val_losses
                        
                        # perform a validation of pure sampling
                        dk_val = masked_components_fft_amplitude(val_batch, val_mask, train_config["max_components"], train_config["masking"], fixed=train_config["fixed_components"])
                        generated_audio_val, steps=sampling_new2(
                            net, val_batch.shape, diffusion_hyperparams, cond=val_batch, mask=val_mask,
                            max_components=train_config["max_components"], dk=dk_val,
                            only_generate_missing=train_config["only_generate_missing"],
                            guidance_weight=1, sampling_with_dk=0, max_components_gen=train_config["max_components"]+1
                        )
                        #diffusion_mse = mean_squared_error(generated_audio_val.cpu().numpy()[~val_mask.cpu().numpy().astype(bool)], val_batch.cpu().numpy()[~val_mask.cpu().numpy().astype(bool)])
                        diffusion_mse = mse_loss(generated_audio_val[~val_mask.bool()], val_batch[~val_mask.bool()])

                        avg_val_loss = val_losses / len(validation_data)
                        print(f"Validation loss: {avg_val_loss}")
                        print(f"Validation diffusion loss: {diffusion_mse}")
                        run.log({ "validation_loss": avg_val_loss},commit=False)
                        run.log({ "validation_diffusion_loss": diffusion_mse},commit=False)
                    net.train()  # Set model back to training mode

            run.log({ "train/epoch": n_iter,"training_loss": loss.item()})

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = f'{n_iter}.pkl'
                checkpoint_name_best = f'{n_iter}_best.pkl'
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),'loss':loss.item()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

                if diffusion_mse.item() < loss_best:
                    loss_best = diffusion_mse.item()
                    #print(loss_best)
                    torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss},
                           os.path.join(output_directory, checkpoint_name_best))
                print(f'Model at iteration {n_iter} is saved')


            n_iter += 1
        epoch_end_time = time.time()  # End timer for the epoch
        if scheduler:
                scheduler.step()
        epoch = epoch + 1
        epoch_duration = epoch_end_time - epoch_start_time
        run.log({ "train_per_epoch": epoch_duration},commit=False)
        # Log the learning rate
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr}, commit=False)
        print(f"Iteration: {n_iter} | Last_Loss: {loss.item()} | Epoch = Epoch Time: {epoch_duration:.2f} sec")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSDS4.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mlspace",
    # Set the wandb project where this run will be logged.
    project="Diffusion",
    # Track hyperparameters and run metadata.
    config=config,
)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams2(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 3:
        model_config = config['wavenet_config']

    train(**train_config)
