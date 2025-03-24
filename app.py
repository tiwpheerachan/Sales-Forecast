# 👉 import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="📊 Forecast Extended", layout="wide")

@st.cache_data
def forecast_next_months(df, months_ahead=3):
    df = df.copy()
    df["month_num"] = pd.to_datetime(df["year_month"]).dt.month
    df["year"] = pd.to_datetime(df["year_month"]).dt.year
    df["month_index"] = df["year"] * 12 + df["month_num"]

    le = LabelEncoder()
    df["brand_enc"] = le.fit_transform(df["brand"])
    df["product_enc"] = le.fit_transform(df["product_name"])
    df["platform_enc"] = le.fit_transform(df["platform"])
    df["campaign_enc"] = le.fit_transform(df["campaign_type"])

    model = GradientBoostingRegressor()
    X = df[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_index"]]
    y = df["sales_thb"]
    model.fit(X, y)

    last_month_index = df["month_index"].max()
    unique_rows = df[["brand_enc", "product_enc", "platform_enc", "campaign_enc",
                      "brand", "product_name", "platform", "campaign_type"]].drop_duplicates()

    forecast_data = []
    for i in range(1, months_ahead+1):
        mi = last_month_index + i
        forecast_month = (datetime(2000, 1, 1) + relativedelta(months=mi)).strftime("%Y-%m")
        for _, row in unique_rows.iterrows():
            forecast_data.append({
                "brand_enc": row["brand_enc"],
                "product_enc": row["product_enc"],
                "platform_enc": row["platform_enc"],
                "campaign_enc": row["campaign_enc"],
                "month_index": mi,
                "brand": row["brand"],
                "product_name": row["product_name"],
                "platform": row["platform"],
                "campaign_type": row["campaign_type"],
                "year_month": forecast_month
            })

    future_df = pd.DataFrame(forecast_data)
    future_df["forecast_sales"] = model.predict(future_df[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_index"]])
    return future_df

# ตัวอย่างการใช้งานร่วมกับข้อมูล summary เดิม
st.header("🔮 Forecast Extension (ล่วงหน้า)")

uploaded_file = st.file_uploader("📂 Upload file again if needed", type=["xlsx"])
months = st.slider("เลือกจำนวนเดือนล่วงหน้าที่ต้องการพยากรณ์", 1, 60, 6)

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    df = xls.parse("Shopee_Product_Perf")
    gmv = xls.parse("GMV_DATA")
    df["Month"] = df["Month"].str.upper()
    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    df["month_num"] = df["Month"].map(month_map)
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["month_num"], day=1))
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    forecast_df = forecast_next_months(df, months)

    st.success(f"✅ คาดการณ์สำเร็จสำหรับ {months} เดือนล่วงหน้า")
    st.dataframe(forecast_df.head(50))

    fig = px.line(forecast_df.groupby("year_month")["forecast_sales"].sum().reset_index(),
                  x="year_month", y="forecast_sales", title="📈 แนวโน้มยอดขายที่คาดการณ์")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 AI แนะนำสินค้า:")
    top_prods = forecast_df.groupby("product_name")["forecast_sales"].sum().sort_values(ascending=False).head(5)
    for name, val in top_prods.items():
        st.markdown(f"**✅ `{name}` คาดว่าจะขายดี (ยอดคาดการณ์: {val:,.0f} THB)** – อาจเป็นเพราะมียอดขายสูงในช่วงก่อนหน้าหรือมีส่วนร่วมใน campaign บ่อยครั้ง")

else:
    st.warning("กรุณาอัปโหลดไฟล์ Excel (.xlsx)")
