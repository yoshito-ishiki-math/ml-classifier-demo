import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.datasets import load_iris

from model_utils import CLASSIFYING_MACHINES
from param_ui import PRAMS_MACHINE
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

pred_class, proba_dict = CLASSIFYING_MACHINES[selected_machine](df, iris, input_df,params)
st.subheader("分類結果")
st.markdown(f"手法「{selected_machine}」によると  \nあなたの入力した花の種類は**{pred_class}**です．")
st.subheader("📊 予測信頼度（各クラスの確率）")
st.json(proba_dict)


#################
##メイン画面
################
