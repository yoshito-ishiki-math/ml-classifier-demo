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



#########################
##前処理
#########################



# --- データ読み込み ---
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['is_input'] = False

#########################
##前処理
#########################



#######
###サイドバー
######
#サイドバー --- ユーザー入力 ---
st.sidebar.header("🧪 特徴量の入力")
st.sidebar.write("あなたの花のパラメーターを入力してね")
user_input = {
    name: st.sidebar.slider(name, float(df[name].min()), float(df[name].max()), float(df[name].mean()))
    for name in iris.feature_names
}
input_df = pd.DataFrame([user_input])
input_df['species'] = 'あなたの入力'
input_df['is_input'] = True

# --- 結合 ---
combined_df = pd.concat([df, input_df], ignore_index=True)

# サイドバー--- 軸の選択 ---
st.sidebar.header("🧭 表示軸の選択")
x_axis = st.sidebar.selectbox("X軸", iris.feature_names, index=2)
y_axis = st.sidebar.selectbox("Y軸", iris.feature_names, index=3)


####
# --- グラフ描画 ---
####
fig = go.Figure()

# 品種ごとに色を割り当て
color_map = {
    "setosa": "blue",
    "versicolor": "green",
    "virginica": "orange"
}

# Iris データを species ごとに描画
for species_name, color in color_map.items():
    species_df = combined_df[(combined_df["species"] == species_name) & (combined_df["is_input"] == False)]
    fig.add_trace(go.Scatter(
        x=species_df[x_axis],
        y=species_df[y_axis],
        mode='markers',
        name=species_name,
        marker=dict(size=4, color=color, opacity=0.6),
        showlegend=True
    ))

# あなたの入力（赤くて大きくて目立つ）
input_only_df = combined_df[combined_df["is_input"] == True]
fig.add_trace(go.Scatter(
    x=input_only_df[x_axis],
    y=input_only_df[y_axis],
    mode='markers+text',
    name='あなたの入力',
    marker=dict(size=12, color='red', symbol='x', line=dict(width=2, color='DarkRed')),
    text=["あなたの入力"],
    textposition='top center',
    showlegend=True
))

# レイアウト調整
fig.update_layout(
    title=f"{x_axis} vs {y_axis} with Your Input",
    xaxis_title=x_axis,
    yaxis_title=y_axis
)


st.sidebar.subheader("以下のプロットは二つの特徴料のプロットとあなたの入力の位置です")
# Streamlitで表示
st.sidebar.plotly_chart(fig, use_container_width=True)
####
# --- グラフ描画 ---
####






#######
###サイドバー
######



#######################
##手法などの記述
####################

###return にはpred_classとproba_dictを入れる．

# --- K-NN 用の関数 ---
def apply_k_nn_method(params):
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
def apply_decision_tree_method(params):
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
def apply_naive_bayes_method(params):
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
def apply_random_forest_method(params):
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
def apply_svm_method(params):
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
def apply_logistic_regression_method(params):
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
def apply_gradient_boosting_method(params):
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
def apply_mlp_method(params):
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

def apply_xgboost_method(params):
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
def apply_lightgbm_method(params):
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



###########
##ハイパラメータUIの定義
##########
def ui_k_nn_params():
    return {
        "n_neighbors": st.slider("近傍数(k)", 1, 15, 5)
    }

def ui_decision_tree_params():
    return {
        "max_depth": st.slider("木の深さ(max_depth)", 1, 10, 3),
        "min_samples_split": st.slider("ノードを分割するために必要な最小のサンプル数(min_samples_split)", 2, 10, 2),
        "random_state": st.slider("乱数シード値(random_state)",0, 100, 42)
    }


def ui_random_forest_params():
    return {
        "n_estimators": st.slider("決定木の本数 (n_estimators)", 10, 300, 100, step=10),
        "max_depth": st.slider("木の最大深さ (max_depth)", 1, 20, 5),
        "min_samples_split": st.slider("ノード分割に必要な最小サンプル数 (min_samples_split)", 2, 20, 2),
        "random_state": st.slider("乱数シード (random_state)", 0, 100, 42)
    }

def ui_svm_params():
    return {
        "C": st.slider("誤差の許容度 (C)", 0.01, 10.0, 1.0, step=0.01),
        "kernel": st.selectbox("カーネル関数 (kernel)", ["linear", "rbf", "poly", "sigmoid"], index=1),
        "gamma": st.selectbox("γ (gamma)", ["scale", "auto"]),
        "random_state": st.slider("乱数シード (random_state)", 0, 100, 42),
        "probability": True  # predict_probaを有効にするため固定
    }

def ui_logistic_regression_params():
    return {
        "C": st.slider("正則化の強さ (C, 小さいほど強い)", 0.01, 10.0, 1.0, step=0.01),
        "max_iter": st.slider("最大イテレーション数 (max_iter)", 100, 1000, 200, step=50),
        "solver": st.selectbox("ソルバー (solver)", ["lbfgs", "liblinear", "saga", "newton-cg"], index=0),
        "random_state": st.slider("乱数シード (random_state)", 0, 1000, 42)
    }


def ui_gradient_boosting_params():
    return {
        "n_estimators": st.slider("決定木の数 (n_estimators)", 10, 500, 100, step=10),
        "learning_rate": st.slider("学習率 (learning_rate)", 0.01, 1.0, 0.1, step=0.01),
        "max_depth": st.slider("各木の深さ (max_depth)", 1, 10, 3),
        "random_state": st.slider("乱数シード (random_state)", 0, 1000, 42)
    }


def ui_mlp_params():
    return {
        "hidden_layer_sizes": (st.slider("隠れ層のユニット数", 5, 200, 10),),
        "activation": st.selectbox("活性化関数 (activation)", ["relu", "tanh", "logistic"], index=0),
        "solver": st.selectbox("最適化アルゴリズム (solver)", ["adam", "sgd", "lbfgs"], index=0),
        "max_iter": st.slider("最大イテレーション数 (max_iter)", 100, 1000, 500, step=50),
        "random_state": st.slider("乱数シード (random_state)", 0, 1000, 42)
    }


def ui_xgboost_params():
    return {
        "n_estimators": st.slider("決定木の本数 (n_estimators)", 10, 500, 100, step=10),
        "learning_rate": st.slider("学習率 (learning_rate)", 0.01, 0.5, 0.1, step=0.01),
        "max_depth": st.slider("木の深さ (max_depth)", 1, 10, 3),
        "subsample": st.slider("サブサンプル比率 (subsample)", 0.5, 1.0, 1.0, step=0.05),
        "colsample_bytree": st.slider("列サンプル比率 (colsample_bytree)", 0.5, 1.0, 1.0, step=0.05),
        "random_state": st.slider("乱数シード (random_state)", 0, 1000, 42)
    }


def ui_lightgbm_params():
    return {
        "n_estimators": st.slider("決定木の本数 (n_estimators)", 10, 500, 100, step=10),
        "learning_rate": st.slider("学習率 (learning_rate)", 0.01, 0.5, 0.1, step=0.01),
        "max_depth": st.slider("木の深さ (max_depth)", 1, 15, -1),  # -1はデフォルト（制限なし）
        "num_leaves": st.slider("葉の数 (num_leaves)", 2, 128, 31),
        "random_state": st.slider("乱数シード (random_state)", 0, 1000, 42)
    }


def ui_naive_bayes_params():
    return {"nb_type": st.write("ガウシアンナイーブベイズです．ハイパラメータはありません．")}


PRAMS_MACHINE = {
    "k-NN": ui_k_nn_params,
    "決定木": ui_decision_tree_params,
    "ロジスティック回帰":ui_logistic_regression_params,
    "ナイーブベイズ":ui_naive_bayes_params,
    "サポートベクターマシン":ui_svm_params,
    "ランダムフォレスト":ui_random_forest_params,
    "ニューラルネット(MLP)":ui_mlp_params,
    "勾配ブースティング":ui_gradient_boosting_params,
    "XGBoost":ui_xgboost_params,
    "LightGBM":ui_lightgbm_params
    }




#################
##メイン画面
################


#タイトル

st.title("Irisデータを教師として,入力した花の分類デモアプリ")

st.write("色々な教師あり学習で分類をします．")





#使用する教師あり学習法を選択

selected_machine = st.radio("使用する手法を選んでね", list(CLASSIFYING_MACHINES.keys()))



st.subheader("ハイパラメータ")

params = PRAMS_MACHINE[selected_machine]()   # パラメータUI表示＆取得

pred_class, proba_dict = CLASSIFYING_MACHINES[selected_machine](params)
st.subheader("分類結果")
st.markdown(f"手法「{selected_machine}」によると  \nあなたの入力した花の種類は**{pred_class}**です．")
st.subheader("📊 予測信頼度（各クラスの確率）")
st.json(proba_dict)


#################
##メイン画面
################
