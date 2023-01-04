# MAMC-Net: Handling class imbalance for improved zero shot domain generalization

Official pytorch code for MAMC-Net 



## Installation
This version has been tested on:
* PyTorch 1.5.1
* python 3.8.3
* cuda 10.2
To convert cumix.yml file into requirements.txt please run:
```
import yaml
with open("environment.yaml") as file_handle:
    environment_data = yaml.load(file_handle)
with open("requirements.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"]:
        package_name, package_version = dependency.split("=")
        file_handle.write("{} == {}".format(package_name, package_version))
```
Once you get requirements.txt,
To install all the dependencies, please run:
```

pip install -r requirements.txt
```
## ZSDG experiments
For setting up the datasets, please download DomainNet from 
[here](http://ai.bu.edu/M3SDA/), using the cleaned version. In the ```data``` folder, you can find the class splits 
(that we defined) and the embeddings used [here](https://www.sciencedirect.com/science/article/pii/S1077314220300928) . To download the data and set up the folder, 
you can also use the script ```download_dnet.sh```:
```
./scripts/download_dnet.sh $DNET_DESIRED_ROOT
```

For reproducing the results, just run the experiments given the corresponding dataset configuration.
For instance, for testing with _painting_ as target:  
```
python main.py --zsl --dg --target painting --config_file configs/zsl+dg/painting.json --data_root $DNET_DESIRED_ROOT --name painting_exps_zsldg or 
bash zsldg.sh
```
you can find other examples in ```scripts/zsldg.sh```. 

## References

If you find this code useful, please cite:
