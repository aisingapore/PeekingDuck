# PeekingDuck Training Pipeline

This is the readme file for the new PeekingDuck Training feature.

Run on default cifar10 dataset:
```sh
python peekingduck/training/main.py
```

To use rsna dataset:
```sh
python ./peekingduck/training/main.py data_module=rsna project_name=rsna
```

To use vegfru dataset:
```sh
python ./peekingduck/training/main.py data_module=vegfru project_name=vegfru
```

To use vegfru5 dataset:
```sh
python ./peekingduck/training/main.py data_module=vegfru5 project_name=vegfru5 debug=True
```

To log in to wandb
```sh
wandb login
```
Copy your key and paste it into your command line when asked to authorize your account.
At the top of your training script, start a new run etc:
```py
wandb.init(project="training-pipeline", entity="peekingduck")
```
