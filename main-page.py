import streamlit as st
import pandas as pd

st.title("My prilo")

lst = ["aa", "bb", "cc"]
d = {"word": lst, "count": [3,4, 160]}
df = pd.DataFrame(data=d)

searched_word = st.text_input(label="Search word")

st.text(body=f"search word: {searched_word}")
st.dataframe(data=df)
