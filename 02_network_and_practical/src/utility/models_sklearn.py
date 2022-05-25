
##############################
# LIBRARIES
##############################

# general
import pandas as pd
import pickle

# data preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# model
from sklearn.svm import LinearSVC

# model evaluation
from sklearn.metrics import classification_report

# time
import time

##############################
# GENERAL CLASS
##############################

class CustomLinearSVC():

    def __init__(self, dataset, language, C):

        # 1. SAVE INFO and PARAMETERS
        self.C = C

        print("> Parameters imported")


        # 2. GET THE SPLITS
        # We can easily do this with the information stored in the 'set' column

        language_col = "text_" + language

        # train
        dataframe_train = dataset.loc[dataset["set"] == "train"]

        # validation 
        dataframe_val = dataset.loc[dataset["set"] == "validation"]

        # test 
        dataframe_test = dataset.loc[dataset["set"] == "test"]

        print("> Dataset correctly divided in training set, validation set and test set")


        # 3. PREPROCESS THE TEXT (with TF-IDF)

        # train
        X_train = dataframe_train[language_col]
        self.Y_train = dataframe_train["labels_new"]

        # validation
        X_val = dataframe_val[language_col]
        self.Y_val = dataframe_val["labels_new"]

        # test
        X_test = dataframe_test[language_col]
        self.Y_test = dataframe_test["labels_new"]

        # TF-IDF setup
        
        # train
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)

        self.__save_count_vectorizer_dict(count_vect, "models/LinearSVC/dictionary/count_vect_tfidf_features.pkl")
        
        tfidf_transformer = TfidfTransformer()
        self.X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        # validation
        loaded_vec = self.__load_count_vectorizer(count_dict_save_path="models/LinearSVC/dictionary/count_vect_tfidf_features.pkl")
        X_val_counts = loaded_vec.fit_transform(X_val)
        tfidf_transformer = TfidfTransformer()
        self.X_val_tfidf = tfidf_transformer.fit_transform(X_val_counts)

        # test
        loaded_vec = self.__load_count_vectorizer(count_dict_save_path="models/LinearSVC/dictionary/count_vect_tfidf_features.pkl")
        X_test_counts = loaded_vec.fit_transform(X_test)
        tfidf_transformer = TfidfTransformer()
        self.X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

        print("> Computed TF-IDF for train, val and test set")

        # 4. INSTANTIATE THE MODEL
        self.MODEL = LinearSVC(C=C)
        print("> Model 'LinearSVC' Instantiated")


    def train_model(self):
        """
        Train the Linear SVC on the given training set
        """

        start = time.time()
        clf = self.MODEL.fit(self.X_train_tfidf, self.Y_train)
        print(f"> Training completed in {round(time.time() - start, 4)} seconds")

        # evaluate the model
        return self.test_model(on="val")


    def test_model(self, on="test"):
        """
        Test the model on the given 'validation' or 'test' set

        Output:
            > classification report (as dictionary) 
        """

        print(f"\n> Testing the model on '{on} set'")

        # set X and Y based on chosen set
        if on == "test":
            X = self.X_test_tfidf
            Y = self.Y_test
        elif on == "val":
            X = self.X_val_tfidf
            Y = self.Y_val
        elif on == "train":
            X = self.X_train_tfidf
            Y = self.Y_train
        
        # predict class labels on given (test) data 
        pred_labels_test = self.MODEL.predict(X)

        # obtain and show model score
        score_test = self.MODEL.score(X, Y)
        print('  - Accuracy Score:', round(score_test,4), "\n")

        # obtain classification report
        print(classification_report(Y, pred_labels_test, zero_division=0, digits=4))

        cr = classification_report(Y, pred_labels_test, zero_division=0, output_dict=True, digits=4)

        return cr


    def __save_count_vectorizer_dict(self, count_vect, save_path):
        """
        Input:
            > count_vect    CountVectorizer() object
            > save_path     path in which the words dictionary will be stored
        """
        pickle.dump(count_vect.vocabulary_,open(save_path,"wb"))


    def __load_count_vectorizer(self, count_dict_save_path):
        """
        Input:
            > count_dict_save_path  dir in which the dict of the count vectorizer is stored
        """
        count_vect_features = pickle.load(open(count_dict_save_path, "rb"))
        loaded_vec = CountVectorizer(vocabulary=count_vect_features)
        return loaded_vec