import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm , get_mask_forecast
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from statistics import mean

from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from statistics import mean

from dtl_utils.datasets_torchloaders import ETTm1TestDataset, ElectricityTestDataset, PTBXLTestDataset

import warnings

warnings.filterwarnings("ignore", message=".*keyword argument dtype in Genred is deprecated.*")


def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_model,
             masking,
             missing_k,
             only_generate_missing):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

            
    # predefine model
    if use_model == 0:
        net = DiffWaveImputer(**model_config).cuda()
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).cuda()
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).cuda()
    else:
        print('Model chosen not available.')
    print_size(net)

    
    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

        
    if "test_size_batch" in trainset_config:
        test_batch_size = trainset_config["test_size_batch"]

        num_samples = test_batch_size
    ### Custom data loading and reshaping ###

    
    if trainset_config['train_data_path'] is not None:    
        if trainset_config['test_data_path'] == "./datasets/Mujoco/test_mujoco.npy":
            testing_data = np.load(trainset_config['test_data_path'])
            testing_data = np.split(testing_data, 4, 0)
            testing_data = np.array(testing_data)
            testing_data = torch.from_numpy(testing_data).float().cuda()
            print('Data loaded',flush=True)
        elif trainset_config['test_data_path'] == "./datasets/ETTm1/test_ettm1_1056.npy":
            ettm1_test_dataset = ETTm1TestDataset(trainset_config)
            testing_data = DataLoader(ettm1_test_dataset, batch_size=500, shuffle=False)
            print('Data loaded',flush=True)
        elif trainset_config['test_data_path'] == "./datasets/Electricity/test_electricity.npy":
            elec_test_dataset = ElectricityTestDataset(trainset_config)
            testing_data = DataLoader(elec_test_dataset, batch_size=num_samples, shuffle=False)
            print('Data loaded',flush=True)
        elif trainset_config['test_data_path'] == "./datasets/PTB-XL/test_ptbxl_1000.npy":
            ptbxl_test_dataset = PTBXLTestDataset(trainset_config)
            testing_data = DataLoader(ptbxl_test_dataset, batch_size=500, shuffle=False)
            print('Data loaded',flush=True)

    all_mse = []
    all_mape = []
    all_mae = []
    all_mape2 = []

    
    for i, batch in enumerate(testing_data):
        
        if not batch.is_cuda:
            batch = batch.to('cuda')
        if masking == 'mnr':
            mask_T = get_mask_mnr(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == 'bm':
            mask_T = get_mask_bm(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == 'rm':
            mask_T = get_mask_rm(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
        
        elif masking == 'forecast':
            mask_T = get_mask_forecast(batch[0],missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        print('mask shape', mask.shape, flush=True)
        print('batch shape', batch.shape,flush=True)
            
        batch = batch.permute(0,2,1)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        sample_length = batch.size(2)
        sample_channels = batch.size(1)
        num_samples = batch.size(0)

        print("batch size 2", batch.size())
        
        generated_audio,steps  = sampling(net, (num_samples, sample_channels, sample_length),
                                   diffusion_hyperparams,
                                   cond=batch,
                                   mask=mask,
                                   only_generate_missing=only_generate_missing)

        end.record()
        torch.cuda.synchronize()

        print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                                                                                             ckpt_iter,
                                                                                             int(start.elapsed_time(
                                                                                                 end) / 1000)))

        
        generated_audio = generated_audio.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy() 
        
        
        outfile = f'imputation{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, generated_audio)

        outfile = f'original{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, batch)

        outfile = f'mask{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, mask)

        print('saved generated samples at iteration %s' % ckpt_iter)
        
        mse = mean_squared_error(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
        mape = mean_absolute_percentage_error(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
        mape2 = mean_absolute_percentage_error(batch[~mask.astype(bool)], generated_audio[~mask.astype(bool)])
        mae = mean_absolute_error(batch[~mask.astype(bool)],generated_audio[~mask.astype(bool)])

        print(f"MSE: {mse:.6f}")
        print(f"MAPE2: {mape:.6f}")

        all_mse.append(mse)
        all_mape.append(mape)
        all_mae.append(mae)
        all_mape2.append(mape2)

    
    outfile = 'results.txt'
    output_file_path = os.path.join(ckpt_path, outfile)

    # Write the metrics to a text file
    with open(output_file_path, 'w') as f:
        f.write(f"MSE: {mean(all_mse):.6f}\n")
        f.write(f"MAPE: {mean(all_mape):.6f}\n")
        f.write(f"MAE: {mean(all_mae):.6f}\n")
        f.write(f"MAPE2 ( the correct one, but make sure everyone follows this one): {mean(all_mape2):.6f}\n")

    print(f"Metrics saved to {output_file_path}")

    def plot_data_with_mask(original, mask, imputation):
        """
        Plot original and imputed data with grey lines indicating masked regions.

        Parameters:
        - original: numpy array of shape (n_samples, n_features, sequence_length)
        - mask: numpy array of shape (n_samples, n_features, sequence_length)
        - imputation: numpy array of shape (n_samples, n_features, sequence_length)
        """
        n_samples = original.shape[0] if isinstance(original, list) else 1

        for i in range(n_samples):
            if n_samples > 1:
                orig_data = original[i]
                mask_data = mask[i]
                imputed_data = imputation[i]
            else:
                orig_data = original
                mask_data = mask
                imputed_data = imputation

            n_features = orig_data.shape[0]

            for j in range(n_features):
                plt.figure(figsize=(10, 5))
                plt.plot(orig_data[j], label='Original')

                for idx, value in enumerate(mask_data[j]):
                    if value == 1:
                        plt.axvline(x=idx, color='lightgrey', linestyle='-')

                plt.plot(imputed_data[j], label='Imputation')

                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title(f'Original vs. Imputed Data (Feature {j})')
                plt.legend()
                
                folder_path = output_directory
                file_name = f'plot_{j}.pdf'

                # Save the figure to the specified folder
                save_path = os.path.join(folder_path, file_name)
                plt.savefig(save_path)


    # Example usage
    randint = np.random.randint(500)
    plot_data_with_mask(batch[randint], mask[randint], generated_audio[randint])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=500,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

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

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
             use_model=train_config["use_model"],
             data_path=trainset_config["test_data_path"],
             masking=train_config["masking"],
             missing_k=train_config["missing_k"],
             only_generate_missing=train_config["only_generate_missing"])
