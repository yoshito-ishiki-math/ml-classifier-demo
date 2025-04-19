import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier




#######
#
######


#######################
##手法などの記述
####################

###return にはpred_classとproba_dictを入れる．

# --- K-NN 用の関数 ---
def apply_k_nn_method(df, iris, input_df, params):
    model = KNeighborsClassifier(**params)
    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]
    model.fit(feature_matrix, species_labels)
    
    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]
    
    proba = model.predict_proba(input_data)[0]  # ← 確率ベクトル（1次元）
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }
    return pred_class, proba_dict



#--決定木
def apply_decision_tree_method(df, iris, input_df,params):
    model = DecisionTreeClassifier(**params)
    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]
    model.fit(feature_matrix, species_labels)

    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]

    proba = model.predict_proba(input_data)[0]
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }

    return pred_class, proba_dict


#--ナイーブベイズ
def apply_naive_bayes_method(df, iris, input_df,params):
    model = GaussianNB()
    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]
    model.fit(feature_matrix, species_labels)

    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]

    proba = model.predict_proba(input_data)[0]
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }

    return pred_class, proba_dict


##ランダムフォレスト
def apply_random_forest_method(df, iris, input_df,params):
    model = RandomForestClassifier(**params)
    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]
    model.fit(feature_matrix, species_labels)

    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]

    proba = model.predict_proba(input_data)[0]
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }

    return pred_class, proba_dict


###SVC(probability=True)
def apply_svm_method(df, iris, input_df,params):
    model = SVC(**params)
    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]
    model.fit(feature_matrix, species_labels)

    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]

    proba = model.predict_proba(input_data)[0]
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }

    return pred_class, proba_dict



###ロジスティック回帰
def apply_logistic_regression_method(df, iris, input_df,params):
    model = LogisticRegression(**params)
    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]
    model.fit(feature_matrix, species_labels)

    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]

    proba = model.predict_proba(input_data)[0]
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }

    return pred_class, proba_dict



#勾配ブースティング
def apply_gradient_boosting_method(df, iris, input_df,params):
    model = GradientBoostingClassifier(**params)
    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]
    model.fit(feature_matrix, species_labels)

    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]

    proba = model.predict_proba(input_data)[0]
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }

    return pred_class, proba_dict


##ニューラルネット(MLP)
def apply_mlp_method(df, iris, input_df,params):
    model = MLPClassifier(**params)

    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]
    model.fit(feature_matrix, species_labels)

    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]

    proba = model.predict_proba(input_data)[0]
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }

    return pred_class, proba_dict



##XGBoost

def apply_xgboost_method(df, iris, input_df,params):
    model = XGBClassifier(**params)
    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]
    
    model.fit(feature_matrix, species_labels)

    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]

    proba = model.predict_proba(input_data)[0]
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }

    return pred_class, proba_dict


#LightGBM
def apply_lightgbm_method(df, iris, input_df,params):
    model = LGBMClassifier(**params)
    feature_matrix = df[iris.feature_names]
    species_labels = iris.target
    input_data = input_df[iris.feature_names]

    model.fit(feature_matrix, species_labels)

    prediction = model.predict(input_data)
    pred_class = iris.target_names[prediction[0]]

    proba = model.predict_proba(input_data)[0]
    proba_dict = {
        name: f"{p:.2%}" for name, p in zip(iris.target_names, proba)
    }

    return pred_class, proba_dict



####
#手法の辞書
####

CLASSIFYING_MACHINES={
    "k-NN":apply_k_nn_method,
    "ロジスティック回帰":apply_logistic_regression_method,
    "ナイーブベイズ":apply_naive_bayes_method,
    "サポートベクターマシン":apply_svm_method,
    "ニューラルネット(MLP)":apply_mlp_method,
    "決定木":apply_decision_tree_method,
    "ランダムフォレスト":apply_random_forest_method,
    "勾配ブースティング":apply_gradient_boosting_method,
    "XGBoost":apply_xgboost_method,
    "LightGBM":apply_lightgbm_method
    }


