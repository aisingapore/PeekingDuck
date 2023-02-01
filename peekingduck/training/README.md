# PeekingDuck Training Pipeline

This is the readme file for the new PeekingDuck Training feature.

```sh
python peekingduck/training/main.py
```

To use rsna dataset
```sh
python ./peekingduck/training/main.py data_module=rsna project_name=rsna
```

To use vegfru dataset
```sh
python ./peekingduck/training/main.py data_module=vegfru project_name=vegfru
```

To use vegfru5 dataset
```sh
python ./peekingduck/training/main.py data_module=vegfru5 project_name=vegfru5 debug=True
```

log in to wandb
```sh
wandb login
```
Copy this key and paste it into your command line when asked to authorize your account.
At the top of your training script, start a new run
```py
wandb.init(project="training-pipeline", entity="peekingduck")
```

Save model inputs and hyperparameters
```py
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}
```

Log gradients and model parameters
```py
wandb.log({"loss": loss})
```