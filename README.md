# Face Recognition baseline

## Train model
To start training model type:
```bash
python3 train.py
```
This method wil use config file './train_config.yml'

To use different config file type:
```bash
python3 train.py --config path_to_file
```
Other optional argumnents:
```
--name-suffix SUFFIX    suffix for saving directories used for storage of training logs and artefacts    
--device DEVICE    override device from configuration file
```

## Results
Overview of results can be found in [evaluation.ipynb](evaluation.ipynb) and in [results](results/)