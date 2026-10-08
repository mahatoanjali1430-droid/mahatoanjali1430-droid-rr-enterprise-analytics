import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Page Config
st.set_page_config(page_title="RR Enterprise Analytics", layout="wide")

# Title
st.title("RR Enterprise - Business Analytics Dashboard")
st.markdown("Industrial Supply Trading Analytics: Customer Churn, Executive Sales, & Inventory Optimization")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("rr_enterprise_sales_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['TotalAmount'] = df['TotalAmount'].fillna(df['Quantity'] * df['UnitPrice'])
    return df

df = load_data()

# Navigation Sidebar
service = st.sidebar.radio(
    "Select Service Module",
    ["Service 1: Customer Churn Analysis", "Service 2: Sales & Billing Dashboard", "Service 3: Inventory ABC Analysis"]
)

# ------------------------------------------------------------------------------
# SERVICE 1: CHURN ANALYSIS
# ------------------------------------------------------------------------------
if service == "Service 1: Customer Churn Analysis":
    st.header("Service 1: B2B Customer Inactivity & Churn Risk")
    
    snapshot_date = df['Date'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'InvoiceID': 'count',
        'TotalAmount': 'sum'
    }).rename(columns={'Date': 'Recency', 'InvoiceID': 'Frequency', 'TotalAmount': 'Monetary'})
    
    rfm['Churn'] = (rfm['Recency'] > 90).astype(int)
    
    col1, col2 = st.columns(2)
    col1.metric("Total Clients", len(rfm))
    col2.metric("At-Risk Clients (>90 Days Inactive)", len(rfm[rfm['Churn'] == 1]))
    
    st.subheader("At-Risk B2B Accounts Needing Immediate Outreach")
    at_risk = rfm[rfm['Churn'] == 1].sort_values(by='Monetary', ascending=False)
    st.dataframe(at_risk.style.format({'Monetary': '₹{:.2f}'}))

# ------------------------------------------------------------------------------
# SERVICE 2: SALES DASHBOARD
# ------------------------------------------------------------------------------
elif service == "Service 2: Sales & Billing Dashboard":
    st.header("Service 2: Executive Sales & Revenue Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Category")
        fig, ax = plt.subplots(figsize=(6, 4))
        cat_rev = df.groupby('Category')['TotalAmount'].sum().sort_values()
        ax.barh(cat_rev.index, cat_rev.values / 1e5, color='#2b5c8f')
        ax.set_xlabel("Revenue (Lakhs ₹)")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Monthly Sales Trend")
        fig, ax = plt.subplots(figsize=(6, 4))
        monthly = df.set_index('Date').resample('ME')['TotalAmount'].sum()
        ax.plot(monthly.index.strftime('%b %Y'), monthly.values / 1e5, marker='o', color='#d9534f')
        plt.xticks(rotation=45)
        ax.set_ylabel("Revenue (Lakhs ₹)")
        st.pyplot(fig)

# ------------------------------------------------------------------------------
# SERVICE 3: INVENTORY ABC ANALYSIS
# ------------------------------------------------------------------------------
elif service == "Service 3: Inventory ABC Analysis":
    st.header("Service 3: Inventory Classification & Dead Stock Risk")
    
    inventory = df.groupby('Category').agg({
        'TotalAmount': 'sum',
        'Quantity': 'sum',
        'InvoiceID': 'count'
    }).rename(columns={'TotalAmount': 'TotalRevenue', 'Quantity': 'UnitsSold', 'InvoiceID': 'OrderFrequency'})
    
    inventory = inventory.sort_values(by='TotalRevenue', ascending=False)
    inventory['CumRevenue'] = inventory['TotalRevenue'].cumsum()
    inventory['CumPct'] = (inventory['CumRevenue'] / inventory['TotalRevenue'].sum()) * 100
    
    def classify_abc(pct):
        if pct <= 70:
            return 'Class A (Fast Moving / High Value)'
        elif pct <= 90:
            return 'Class B (Medium Value)'
        else:
            return 'Class C (Slow Moving / Dead Stock Risk)'
            
    inventory['ABC_Category'] = inventory['CumPct'].apply(classify_abc)
    st.dataframe(inventory[['TotalRevenue', 'UnitsSold', 'OrderFrequency', 'ABC_Category']].style.format({'TotalRevenue': '₹{:.2f}'}))
