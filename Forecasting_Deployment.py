from scipy.stats import boxcox 
from sklearn import preprocessing
import statsmodels.api as sm
import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

st.title("Forecasting Brents Oil Prices")
st.sidebar.header('User Inputs :')
days = st.sidebar.number_input("Days Forecast")
st.write(' Mentors: Vinod and Deepika')
st.write('This will Forcast Brent Oil Prices for next days')
st.write('Team P80-3: Thakur Aashish, Sanket , Dalmiya , Rooba , Hariharan, Nikhil')
if days == 0:
    new_title = '<p style="font-family:sans-serif; color:RED; font-size: 22px;">PLEASE ENTER NO OF DAYS GREATER THAN 0</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
else:
    oil=pd.read_excel('Brent.xls', index_col=[0],parse_dates = [0])
    ### Treating outlier 
    for i in oil:
        oil[i],lmbda=boxcox(oil[i], lmbda=None)
    #STationarity
    oil['Spotprice_First_differencing']= oil['SpotPrice'] - oil['SpotPrice'].shift(1)
    oil['Spotprice_First_differencing']=oil['Spotprice_First_differencing'].dropna()
    # train test
    train_ar = oil.head(1083)
    test_ar = oil.tail(226)


    ## Sarima
    import statsmodels.api as sm
    SARIMAmodel = sm.tsa.statespace.SARIMAX(train_ar['Spotprice_First_differencing'], order=(0,1,0),seasonal_order = (3,1,3,5))
    SARIMA_model_fit = SARIMAmodel.fit()


    SARIMA_fore = SARIMA_model_fit.forecast(steps=int(days)) 

    SARIMA_fore=pd.DataFrame(SARIMA_fore)
    SARIMA_fore.columns=[('forcast')]

    st.write(SARIMA_fore)
    ##plot
    st.line_chart(data=SARIMA_fore, width=0, height=0, use_container_width=True)

