![FFT-DM](example_diffusion_process.gif)


## Install the requiered libraries 
```
pip install -r requirements.txt
```
or 


## Datasets

* You can download and store them with the next command
```
python3 get_data.py
```


## How to use the models with the decomposable diffusion process

Choose the config files from the config folder.

Choose from any combination of model and dataset


# Fast experiment - ettm1 dataset 192-forecasting
(example using Diffwave, you can choose which model)
```
python3 train.py -c config/config_DiffWave_ettm1_FDM.json
python3 inference.py -c config/config/config_DiffWave_ettm1_FDM.json
```


## Use your own data

Create your own config file with a new train_data_path, and, if necessary,
adapt the BaseTrainingDataset class to your needs:

```
class BaseTrainingDataset(Dataset):
    def __init__(self, trainset_config):
        
        training_data = np.load(trainset_config['train_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float()

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
```

and add it to the list 