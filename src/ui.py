import streamlit as st
import os


def load_css():

    css_path = os.path.join(
        "assets",
        "style.css"
    )

    if os.path.exists(css_path):

        with open(
            css_path,
            "r",
            encoding="utf-8"
        ) as file:

            css = file.read()

        st.markdown(
            f"""
            <style>
            {css}
            </style>
            """,
            unsafe_allow_html=True
        )