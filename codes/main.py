import yaml
import argparse
from utils import LoadData, ImgDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import Vanilla_CNN, FC_Network, CNN_classifier, Vanilla_CNN_no_dropout, Vanilla_CNN_no_BN, CNN_classifier_no_dropout, \
                  CNN_classifier_no_BN, CNN_classifier_60, Vanilla_CNN_60
from torchinfo import summary
from train import Trainer
import torch
import torch.nn as nn
from torch.optim import Adam

def train_test(config):

    num_classes = config['num_classes']
    num_samples_train = config['num_samples_train']
    num_samples_test = config['num_samples_test']
    random_seed = config['random_seed']
    data_folder = config['data_folder']
    train_image, train_label, test_image, test_label = LoadData(num_classes, 
                                                                num_samples_train, 
                                                                num_samples_test, 
                                                                random_seed, 
                                                                data_folder)

    x_train, x_val, y_train, y_val = train_test_split(train_image, train_label, 
                                                  test_size=0.1, random_state=2022)
    
    batch_size = config['batch_size']
    train_data = ImgDataset(x_train.reshape(-1,1,28,28), y_train)
    val_data = ImgDataset(x_val.reshape(-1,1,28,28), y_val)
    test_data = ImgDataset(test_image.reshape(-1,1,28,28), test_label)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


    epochs = config['epochs']
    model_name = config['model_name']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    model_dir = config['model_dir']
    dropout = config['dropout']
    bn = config['batch_normalization']
    if model_name == 'fc':
        model = FC_Network()
    elif model_name == 'cnn':
        if num_classes == 50 and dropout == True and bn == True:
            model = Vanilla_CNN()
        elif num_classes == 60:
            model = Vanilla_CNN_60()
            model_name = model_name + '_60'
        elif num_classes == 50 and dropout == False:
            model = Vanilla_CNN_no_dropout()
            model_name = model_name + '_no_dropout'
        elif num_classes == 50 and bn == False:
            model = Vanilla_CNN_no_BN()
            model_name = model_name + '_no_BN'
        else:
            print("No implemented model, use the default model")
            model = Vanilla_CNN()
    elif model_name == 'new_cnn':
        if num_classes == 50 and dropout == True and bn == True:
            model = CNN_classifier()
        elif num_classes == 60:
            model = CNN_classifier_60()
            model_name = model_name + '_60'
        elif num_classes == 50 and dropout == False:
            model = CNN_classifier_no_dropout()
            model_name = model_name + '_no_dropout'
        elif num_classes == 50 and bn == False:
            model = CNN_classifier_no_BN()
            model_name = model_name + '_no_BN'
        else:
            print("No implemented model, use the default model")
            model = CNN_classifier()
    summary(model)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trainer = Trainer(model=model, 
                      epochs=epochs, 
                      train_dataloader=train_loader, 
                      val_dataloader=val_loader, 
                      test_dataloader=test_loader,
                      criterion=criterion, 
                      optimizer=optimizer, 
                      lr=learning_rate, 
                      device=device, 
                      model_dir=model_dir, 
                      model_name=model_name, 
                      random_seed=random_seed)
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the config', default='./test/config_trfl.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    train_test(config)


