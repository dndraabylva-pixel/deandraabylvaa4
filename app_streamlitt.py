import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Belajar Klasifikasi Kentang",
	page_icon = ":potato:"
)

model = joblib.load("model_klasifikasi_kentang.joblib")

st.title(":potato: Belajar Klasifikasi Kentang")
st.markdown("Aplikasi machine learning classification untuk memprediksi kualitas kentang")

berat_gram = st.slider("Berat gram", 65.0, 350.0, 200.0)
diameter_cm = st.slider("Diameter", 5.0, 10.0, 6.50)
kadar_air_persen = st.slider("Kadar air persen", 70.0, 90.0, 77.0)
jenis_kentang = st.pills("Jenis kentang", ["Granola", "Atlantik", "Russet", "kentang_merah"], default="Granola")
metode_panen = st.pills("Metode panen", ["Mesin", "Manual"], default= "Mesin")
kondisi_penyimpanan = st.pills("Kondisi Penyimpanan", ["Gudang", "Suhu Ruang", "Cold Storage"], default="Gudang")
ada_tunas = st.pills("Ada tunas", ["Tidak", "Ya"], default="Tidak")
tingkat_kematangan = st.pills("Tingkat Kematangan", ["Matang", "Belum Matang", "Sangat Matang"], default="Matang")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[berat_gram,diameter_cm,kadar_air_persen,jenis_kentang,metode_panen,kondisi_penyimpanan,ada_tunas,tingkat_kematangan]], columns=["berat_gram","diameter_cm","kadar_air_persen","jenis_kentang","metode_panen","kondisi_penyimpanan","ada_tunas","tingkat_kematangan"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi **{prediksi}**dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()
st.divider()
st.caption("Dibuat oleh **bylva** :cat:")