import streamlit as st

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
    st.write("### 💡 MLP（多層パーセプトロン）のパラメータ設定")

    num_layers = st.number_input("層の数", min_value=1, max_value=5, value=1, step=1)
    
    hidden_layer_sizes = []
    for i in range(num_layers):
        units = st.slider(f"第{i+1}層のユニット数", 5, 200, 10, key=f"layer_{i}")
        hidden_layer_sizes.append(units)

    return {
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
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
    {"nb_type": st.write("ガウシアンナイーブベイズです．ハイパラメータはありません．")}
    return None


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


