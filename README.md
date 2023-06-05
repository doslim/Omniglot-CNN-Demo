# Omniglot-CNN-Demo
 
![](https://visitor-badge.glitch.me/badge?page_id=Doslim.Omniglot-CNN-Demo)

In this repository, we implement several deep neural networks for the [Omniglot challenge](https://github.com/brendenlake/omniglot). 

## Requirements and Structures

The required environments are as follows.

- torch-1.11.0
- torchinfo-1.7.1
- scikit-learn-1.2.0
- Pillow-8.4.0

The structure of our project is as follows.
- codes: contain all the codes.
  - main.py: the entrance of our project.
  - train.py: define the class to train the model.
  - model.py: define all the models.
  - utils.py: load the data for training and evaluations.
  - config.yaml: store the configurations for model training.
- demo.ipynb: a jupyter notebook file that shows how our codes work.
- omniglot_resized: contain the data.
- output: the path to save models and evaluation results.
- report.pdf: a brief report of details of our implementations, experimental results and analysis.

## Usage
To run our codes, change to the codes directory and use the following command
```
python main.py --config = config.yaml
```

