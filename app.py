import os
import sys
import subprocess
import time
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dependencies are installed by Streamlit Cloud from requirements.txt.
# For local Windows setup, run setup_local.py once.
if __name__ == "__main__" and not os.environ.get("STREAMLIT_APP_RUNNING"):
    env = os.environ.copy()
    env["STREAMLIT_APP_RUNNING"] = "1"
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)], env=env, check=True)
    sys.exit(0)

st.set_page_config(
    page_title="AI Stock Market",
    page_icon="📈",
    layout="wide"
)

# PRELOADER
if "preloader_complete" not in st.session_state:
    with st.spinner("Loading AI Stock Market..."):
        time.sleep(2)
    st.session_state.preloader_complete = True

# PUBLIC DEVELOPMENT SERVICES
from urllib.parse import quote

CONTACT_EMAIL = "jinesh9676@gmail.com"
CONTACT_PHONE = "+91 8799428255"
FULL_NAME = "Jinesh Paras Shah"


def show_development_services():
    st.header("💼 Hire a Python AI Developer")
    st.write("Custom Python, Machine Learning, Streamlit and AI-powered applications for businesses, institutes and individuals. Remote projects accepted across India.")

    services = [
        ("Python Automation", "₹1,500+", "Automation scripts, data processing and reports"),
        ("Streamlit Dashboard", "₹3,000+", "Interactive dashboards and data visualization"),
        ("Stock Market Analytics", "₹5,000+", "Historical charts and technical analysis tools"),
        ("Machine Learning", "₹7,500+", "Model development, evaluation and integration"),
        ("LSTM AI Application", "₹10,000+", "Time-series forecasting and model diagnostics"),
        ("Custom Python AI Software", "₹15,000+", "End-to-end development and deployment"),
    ]
    for i in range(0, len(services), 3):
        for col, (name, price, details) in zip(st.columns(3), services[i:i + 3]):
            with col:
                with st.container(border=True):
                    st.subheader(name)
                    st.metric("Starting Price", price)
                    st.write(details)

    st.info("Prices are indicative. Final quotations depend on requirements, complexity and timeline.")
    st.subheader("📩 Request a Custom Project")
    with st.form("client_project_enquiry"):
        client_name = st.text_input("Full Name", max_chars=100)
        client_email = st.text_input("Email Address", max_chars=150)
        client_phone = st.text_input("Phone / WhatsApp Number", max_chars=20)
        project_type = st.selectbox("Project Type", [x[0] for x in services])
        budget = st.selectbox("Estimated Budget", ["Below ₹3,000", "₹3,000 - ₹5,000", "₹5,000 - ₹10,000", "₹10,000 - ₹25,000", "Above ₹25,000", "Discuss Budget"])
        project_details = st.text_area("Project Requirements", max_chars=3000, height=150)
        submitted = st.form_submit_button("Prepare Project Enquiry", type="primary", use_container_width=True)

    if submitted:
        if not client_name.strip():
            st.error("Please enter your name.")
        elif "@" not in client_email or "." not in client_email.rsplit("@", 1)[-1]:
            st.error("Please enter a valid email address.")
        elif not project_details.strip():
            st.error("Please describe your project.")
        else:
            subject = f"Python AI Project Enquiry - {project_type}"
            message = (f"New Python AI Development Enquiry\n\nClient Name: {client_name}\nEmail: {client_email}\nPhone: {client_phone}\n\nProject Type: {project_type}\nEstimated Budget: {budget}\n\nProject Requirements:\n{project_details}\n\nSource: AI Stock Market Streamlit Website")
            url = f"mailto:{CONTACT_EMAIL}?subject={quote(subject)}&body={quote(message)}"
            st.success("Your enquiry is ready. Click below to open your email application and send it.")
            st.link_button("📧 Send Project Enquiry", url, type="primary", use_container_width=True)

    st.divider()
    st.subheader("📞 Contact for Development Services")
    c1, c2, c3 = st.columns(3)
    c1.write(f"📧 Email: {CONTACT_EMAIL}")
    c2.write(f"📱 Phone: {CONTACT_PHONE}")
    c3.write(f"👨‍💻 Developer: {FULL_NAME}")
    st.link_button("💬 Contact Developer on WhatsApp", "https://wa.me/918799428255?text=" + quote("Hello, I want to discuss a custom Python AI development project."), use_container_width=True)
    st.caption("Python Development • AI / ML • Data Analytics • Streamlit Applications")
