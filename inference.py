import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm , get_mask_forecast,get_mask
from utils.util import find_max_epoch, print_size, sampling_new2, calc_diffusion_hyperparams,calc_diffusion_hyperparams2,calc_quantile_CRPS,calc_sample_CRPS
from utils.util_plots import plot_diffusion_steps_gif
import utils.util as util
from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer
from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from statistics import mean

from dtl_utils.datasets_torchloaders import ETTm1TestDataset, ElectricityTestDataset, PTBXLTestDataset,Synth1TestDataset

import warnings

warnings.filterwarnings("ignore", message=".*keyword argument dtype in Genred is deprecated.*")


def generate(output_directory,
             num_obs,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_model,
             masking,
             missing_k,
             only_generate_missing,
             max_components,
             max_components_gen,
             sampling_with_dk,
             fixed_arguments):
    
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
    local_path = "T{}_beta0{}_betaT{}_n{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"],max_components)

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

    if sampling_with_dk == None:
        print('sampling_with_dk is none')
        sampling_with_dk = 1

            
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

    if fixed_arguments == 0:
        diffusion_hyperparams['fixed'] = False
    else:
        diffusion_hyperparams['fixed'] = True

    
    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        max_value = max([int(f.split('_')[0]) for f in os.listdir(ckpt_path) if f.endswith('best.pkl')])
        ckpt_iter = str(max_value) + '_best'    
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        print(model_path)
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
            print('Data loaded')
        elif trainset_config['test_data_path'] == "./datasets/ETTm1/test_ettm1_1056.npy":
            ettm1_test_dataset = ETTm1TestDataset(trainset_config)
            testing_data = DataLoader(ettm1_test_dataset, batch_size=500, shuffle=False)
            print('Data loaded')
        elif trainset_config['test_data_path'] == "./datasets/Electricity/test_electricity.npy":
            ucr_test_dataset = ElectricityTestDataset(trainset_config)
            testing_data = DataLoader(ucr_test_dataset, batch_size=500, shuffle=False)
            print('Data loaded')
        elif trainset_config['test_data_path'] == "./datasets/PTB-XL/test_ptbxl_1000.npy":
            ptbxl_test_dataset = PTBXLTestDataset(trainset_config)
            testing_data = DataLoader(ptbxl_test_dataset, batch_size=500, shuffle=False)
            print('Data loaded')
        elif trainset_config['test_data_path'] == "./datasets/synth1/y_test.npy":
            dataset = Synth1TestDataset(trainset_config)
            testing_data = DataLoader(dataset, batch_size=1000, shuffle=False)
            print('Data loaded')
        else:
            print('bro nao fez load do dataset')

    all_mse = []
    all_mape = []
    all_mae = []
    all_mape2 = []
    all_crps = []
    all_crps2 = []
    print('Starting generation')

    
    net.eval()
    with torch.no_grad():
        #generated_audio_samples = []
        for i,batch in enumerate(testing_data):
            generated_audio_samples = []
            if not batch.is_cuda:
                    batch = batch.to('cuda')
                
            mask,loss_mask = get_mask(batch, masking, missing_k)
            print("batch size", batch.size())
            #dk = util.masked_components_fft_amplitude(batch, mask,max_components,masking,fixed=fixed_arguments)

            
            batch = batch.permute(0,2,1)  # Define mask generation separately

            dk = util.masked_components_fft_amplitude(batch, mask,max_components,masking,fixed=fixed_arguments)
            for j in range(num_obs):
                print(j,"/",num_obs,"number of observations",flush=True)

                #start = torch.cuda.Event(enable_timing=True)
                #end = torch.cuda.Event(enable_timing=True)
                #start.record()

                sample_length = batch.size(2)
                sample_channels = batch.size(1)
                batch_len = batch.size(0)

                generated_audio,_  = sampling_new2(net, (batch_len, sample_channels, sample_length),
                                        diffusion_hyperparams,
                                        cond=batch,
                                        mask=mask,
                                        max_components=max_components,
                                        dk=dk,
                                        only_generate_missing=only_generate_missing,guidance_weight=1,sampling_with_dk=sampling_with_dk,max_components_gen=max_components_gen)

                #end.record()

                #plot_diffusion_steps_gif(steps,batch.detach().cpu().numpy(),save_path="diffusion_process.gif")

                #torch.cuda.synchronize()

                #print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                #                                                                                    ckpt_iter,
                #                                                                                    int(start.elapsed_time(
                #                                                                                        end) / 1000)))

                
                generated_audio = generated_audio.detach().cpu().numpy()
                #batch = batch.detach().cpu().numpy()
                #mask = mask.detach().cpu().numpy() 
                
                


                #print(f"Batch {i} MSE: {mse:.6f}")
                #print(f"Batch {i} MAPE: {mape:.6f}")
                generated_audio_samples.append(generated_audio)          
            torch_generated_audio_samples = torch.Tensor(np.array(generated_audio_samples))
            crps = calc_quantile_CRPS(batch, torch_generated_audio_samples,loss_mask)
            crps_sample = calc_sample_CRPS(batch, torch_generated_audio_samples,loss_mask)
            median_generated_audio = torch.quantile(torch_generated_audio_samples,0.5,dim=0)
            median_generated_audio = median_generated_audio.detach().cpu().numpy()
            batch = batch.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            mse = mean_squared_error(median_generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
            mape = mean_absolute_percentage_error(median_generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
            mae = mean_absolute_error(batch[~mask.astype(bool)],median_generated_audio[~mask.astype(bool)])
            mape2 = mean_absolute_percentage_error(batch[~mask.astype(bool)], median_generated_audio[~mask.astype(bool)])

            all_mse.append(mse)
            all_mape.append(mape)
            all_mae.append(mae)
            all_mape2.append(mape2)
            all_crps.append(crps)
            all_crps2.append(crps_sample)

            print(f"Batch {i} MSE: {mse:.6f}")
            print(f"Batch {i} MAPE: {mape:.6f}")


            outfile = f'imputation{i}.npy'
            new_out = os.path.join(ckpt_path, outfile)
            np.save(new_out, generated_audio_samples)

            outfile = f'original{i}.npy'
            new_out = os.path.join(ckpt_path, outfile)
            np.save(new_out, batch)

            outfile = f'mask{i}.npy'
            new_out = os.path.join(ckpt_path, outfile)
            np.save(new_out, mask)

            #print('saved generated samples at iteration %s' % ckpt_iter)

            #print(f"Batch {i} MSE: {mse:.6f}")
            #print(f"Batch {i} MAPE: {mape:.6f}")

            outfile = 'results.txt'
            output_file_path = os.path.join(ckpt_path, outfile)

            # Write the metrics to a text file
    with open(output_file_path, 'w') as f:
        f.write(f"MSE: {mean(all_mse):.6f}\n")
        f.write(f"MAPE: {mean(all_mape):.6f}\n")
        f.write(f"MAPE: {mean(all_mae):.6f}\n")
        f.write(f"MAPE ( this is the correct one, but make sure): {mean(all_mape2):.6f}\n")
        f.write(f"CRPS: {mean(all_crps):.6f}\n")
        f.write(f"CRPS sample: {mean(all_crps2):.6f}\n")

    print(f"Metrics saved to {output_file_path}")
    
        
    
    print('Total MSE:', mean(all_mse))

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
    #randint = np.random.randint(0, len(batch))
    #plot_data_with_mask(batch[randint], mask[randint], generated_audio[randint])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=500,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*keyword argument dtype in Genred is deprecated.*")


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
    diffusion_hyperparams = calc_diffusion_hyperparams2(
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
             only_generate_missing=train_config["only_generate_missing"],
             max_components=train_config["max_components"],
             fixed_arguments=train_config["fixed_components"],
             )
