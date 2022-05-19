
##############################
# LIBRARIES
##############################

# general
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

# dataset
from torch.utils.data import Dataset, DataLoader

# pytorch
import torch
import torch.nn.functional as F
import torch.nn as nn

# metrics
from sklearn.metrics import mean_squared_error

# time
import time


##############################
# GENERAL CLASS
##############################

class PytorchModel():

    def __init__(
        self, 
        model_type, 
        dataset, 
        language, 
        device,
        batch_size, 
        epochs, 
        vocab_size,
        embedding_dim, 
        hidden_dim,
        learning_rate,
        dropout_p
        ) -> None:

        # measure init time
        init_start = time.time()

        # 1. SAVE INFO and HYPERPARAMETERS

        self.MODEL_TYPE = model_type
        self.DEVICE = device
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        
        self.VOCAB_SIZE = vocab_size        
        self.EMBEDDING_DIM = embedding_dim
        self.HIDDEN_DIM = hidden_dim
        self.LEARNING_RATE = learning_rate
        self.DROPOUT_P = dropout_p
        self.NUM_CLASSES = len(set(dataset["labels_new"])) # find number of classes (using the whole dataset)

        print("> Parameters imported")

        # 2. GET THE SPLITS
        # We can easily do this with the information stored in the 'set' column

        language_col = "text_" + language + "_enc"

        # train 
        dataframe_train = dataset.loc[dataset["set"] == "train"]
        self.X_TRAIN = list(dataframe_train[language_col])
        self.Y_TRAIN = list(dataframe_train["labels_new"])

        # validation 
        dataframe_val = dataset.loc[dataset["set"] == "validation"]
        self.X_VAL = list(dataframe_val[language_col])
        self.Y_VAL = list(dataframe_val["labels_new"])

        # test 
        dataframe_test = dataset.loc[dataset["set"] == "test"]
        self.X_TEST = list(dataframe_test[language_col])
        self.Y_TEST = list(dataframe_test["labels_new"])

        print("> Dataset correctly divided in training set, validation set and test set")


        # 3. CREATE PYTORCH DATASETS and DATA LOADERS

        self.TRAIN_DATASET = TextsDataset(self.X_TRAIN, self.Y_TRAIN)
        self.VAL_DATASET = TextsDataset(self.X_VAL, self.Y_VAL)
        self.TEST_DATASET = TextsDataset(self.X_TEST, self.Y_TEST)
        
        self.TRAIN_DL = DataLoader(self.TRAIN_DATASET, batch_size=self.BATCH_SIZE, shuffle=True)
        self.VAL_DL = DataLoader(self.VAL_DATASET, batch_size=self.BATCH_SIZE)
        self.TEST_DL = DataLoader(self.TEST_DATASET, batch_size=self.BATCH_SIZE)

        print("> Created Pytorch datasets and dataloaders")

        # 4. INSTANTIATE THE REQUIRED MODEL

        if self.MODEL_TYPE == "LSTM":

            self.MODEL = LSTM_fixed_len(
                vocab_size=self.VOCAB_SIZE, 
                embedding_dim=self.EMBEDDING_DIM, 
                hidden_dim=self.HIDDEN_DIM,
                num_classes=self.NUM_CLASSES,
                dropout=self.DROPOUT_P,
                device=self.DEVICE
            ).to(self.DEVICE)


            print(f"> Model '{self.MODEL_TYPE}' instantiated")
        else:
            print(f"> Error: model '{self.MODEL_TYPE}' not available. Please choose between ['LSTM' | ]")


        # print execution time
        print(f"> Initialization required {round(time.time() - init_start, 4)} seconds")



    def train_model(self):
        """
        Training function

        Returns
        -------
        dataframe
            with training results (obtained on both training and validation sets)
        """

        print("=========================================")
        print(f"> Training Started")
        print(f"  - Total Epochs: {self.EPOCHS}")
        print("=========================================")

        # setup loss and optimizers
        parameters = filter(lambda p: p.requires_grad, self.MODEL.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.LEARNING_RATE)

        # create placeholder for results
        df_epoch = []
        df_training_loss = []
        df_validation_loss = []
        df_validation_accuracy = []

        # for each epoch...
        for i in range(self.EPOCHS):

            # training mode
            self.MODEL.train()

            # reset loss and total
            sum_loss = 0.0
            total = 0

            custom_desc = f"> Epoch {i+1}"
            for x, y, l in tqdm(self.TRAIN_DL, desc=custom_desc):

                # input preprocess
                x = x.long().to(self.DEVICE)
                y = y.long().to(self.DEVICE)

                # forward pass
                y_pred = self.MODEL(x, l)
                optimizer.zero_grad()
                loss = F.cross_entropy(y_pred, y)

                # backwards pass
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()*y.shape[0]
                total += y.shape[0]

            # evaluate the model on the validation set
            val_loss, val_acc, classes_accuracy = self.__validation_metrics()

            # display results
            print(f" - Training Loss        {round(sum_loss/total, 4)}")
            print(f" - Validation Loss      {round(val_loss, 4)}")
            print(f" - Validation Accuracy  {round(float(val_acc), 4)}")

            print(" - Validation Accuracy (per class)")
            for key, value in classes_accuracy.items():

                temp_corr = value['correct']
                temp_total = value['total']
                temp_class_acc = round(temp_corr/temp_total, 4)
                value['accuracy'] = temp_class_acc
                value['epoch'] = i+1

                print(f"   * Class {key}\t {temp_class_acc} [{temp_corr} out of {temp_total}]")
            print("\n=========================================")

            # update list results
            df_epoch.append(i+1)
            df_training_loss.append(round(sum_loss/total, 4))
            df_validation_loss.append(round(val_loss, 4))
            df_validation_accuracy.append(round(float(val_acc), 4))

            if i == 0:
                classes_res_df = pd.DataFrame.from_dict(classes_accuracy, orient="index")
            else:
                classes_res_df = pd.concat([classes_res_df, pd.DataFrame.from_dict(classes_accuracy, orient="index")])

        # create df results and return it

        custom_zip = zip(
            df_epoch, 
            df_training_loss, 
            df_validation_loss, 
            df_validation_accuracy
            )

        global_res_df = pd.DataFrame(
            list(custom_zip),
            columns = ["epoch", "training_loss", "validation_loss", "validation_accuracy"]
            )
        
        return global_res_df, classes_res_df


    def __validation_metrics(self):
        """
        Evaluates a model performance

        Returns
        -------
        val_loss, val_acc, classes_accuracy
        """

        # evaluation mode (no need for backwards propagation)
        self.MODEL.eval()

        # setup placeholders
        correct = 0
        total = 0
        sum_loss = 0.0
        classes_accuracy = self.__get_classes_accuracy_dict(n_classes=self.NUM_CLASSES)

        for x, y, l in self.VAL_DL:

            # input preprocess
            x = x.long().to(self.DEVICE)
            y = y.long().to(self.DEVICE)

            # obtain predictions
            y_hat = self.MODEL(x, l)
            loss = F.cross_entropy(y_hat, y)
            pred = torch.max(y_hat, 1)[1]

            # obtain results

            # accuracy of each class 
            for prediction, ground_truth in zip(pred, y):

                # update correct if the prediction is correct
                if int(prediction) == int(ground_truth):
                    classes_accuracy[int(ground_truth)]["correct"] +=1
                
                # update total number of texts in current class
                classes_accuracy[int(ground_truth)]["total"] +=1

            # global accuracy and loss
            correct += (pred == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]

        return sum_loss/total, correct/total, classes_accuracy


    def __get_classes_accuracy_dict(self, n_classes):

        solution_dict = {}

        for i in range(n_classes):
            solution_dict[i] = {"correct":0,"total":0}

        return solution_dict


##############################
# DATASET CLASS
##############################

class TextsDataset(Dataset):
    """
    Class to handle datasets in Pytorch
    """
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]


#################################
# LSTM (with fixed length) CLASS
#################################

class LSTM_fixed_len(torch.nn.Module):
    """
    LSTM model (with fixed length)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout, device):

        # SETUP
        self.DEVICE = device

        # calls the constructor of the parent (nn.Module) 
        # --> so that any initialization done in the super class is still done
        super().__init__()

        # LAYERS

        # Embedding Layer: a simple lookup table that stores embeddings of a fixed dictionary and size.
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,      # size of the dictionary of embeddings
            embedding_dim=embedding_dim,    # the size of each embedding vector
            padding_idx=0                   # the entries at padding_idx do not contribute to the gradient
            )

        # LSTM: applies a multi-layer long short-term memory (LSTM) RNN to an input sequence
        self.lstm = nn.LSTM(
            input_size=embedding_dim,       # the number of expected features in the input x
            hidden_size=hidden_dim,         # the number of features in the hidden state h
            batch_first=True                # input and output tensors provided as (batch, seq, feature)
            )

        # Linear Layer: applies a linear transformation to the incoming data 
        self.linear = nn.Linear(
            in_features = hidden_dim,       # size of each input sample
            out_features = num_classes      # size of each output sample
            )
        
        # Dropout Layer: during training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
        self.dropout = nn.Dropout(
            p=dropout                       # probability of an element to be zeroed
            )
        
    def forward(self, x, l):

        # Set initial hidden and cell states 
        x = self.embeddings(x).to(self.DEVICE)
        x = self.dropout(x).to(self.DEVICE)

        # Forward propagate LSTM
        lstm_out, (ht, ct) = self.lstm(x)

        # Decode the hidden state of the last time step and return
        return self.linear(ht[-1])





