import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import altair as alt
from datetime import datetime
from pandas import Series

st.title("Open Data Forecasting")

#sidebar
st.sidebar.title("Predicting the Population")

#description on the mainpage
st.markdown("Using the census data of 🇵🇰 - we predict the future population of the country ")

#copying it to the sidebar
st.sidebar.markdown("Using the census data of 🇵🇰 - we predict the population of the country in the coming years  ")

#loading the csv file
#data_url = ("C:/Users/hp/Desktop/Python dashboards/streamlit/combined_districts.csv")

DATA_URL = ("combined_districts.csv")

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    #data['Year'] = pd.to_datetime(data['Year']).dt.strftime('%Y')
    return data
df = load_data()


#metrics =['district','children','teen','young_adult','adult','senior','total']
#cols = st.selectbox('Population metric to view', metrics)
# let's ask the user which column should be used as Index
#if cols in metrics:   
 #   metric_to_show_in_pop_Layer = cols

# Filters UI
# subset_data = df
# province_name_input = st.sidebar.multiselect(
# 'Province',
# df.groupby('province').count().reset_index()['province'].tolist())
# # by country name
# if len(province_name_input) > 0:
#     subset_data = df[df['province'].isin(province_name_input)]

# Create a list of possible values and multiselect menu with them in it.
# provinces = df['province'].unique()
# provinces_selected = st.sidebar.multiselect('Select provinces',provinces)

# # Mask to filter dataframe
# mask_provinces = df['province'].isin(provinces_selected)

# option1 = df[mask_provinces]

st.sidebar.markdown("### Predicted population per year")
st.markdown("### Actual Population in different Years")
sum = df.groupby(['year'])['Total'].sum().reset_index()
fig = px.bar(sum, x='year', y='Total', color='year')
st.plotly_chart(fig)

#population_count = df.groupby(['province'])['Total'].agg('sum')
#population_count = df.groupby(['province','district'])['Total'].sum()
#population_count = df.groupby(['province'])['Total'].sum().reset_index()
#population_count = df['province'].value_counts()
#population_count = pd.DataFrame({'province':population_count.index, 'Total':population_count.values})
option = st.sidebar.selectbox(
        'Please Select the Year you want prediction for',
        (2007, 2009, 2011, 2013, 2015))

option1 = st.sidebar.selectbox(
        'Please Select the Province you want prediction for',
        ("Sindh", "Balochistan", "Punjab", "NWFP"))


#graph 2 
st.subheader('Overall prediction of Pakistan By Our Model')
#code for overall
def model(year):
    if (year == 2007):
        data = pd.read_csv('category_overall/2007_group_file.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        sum = df.loc[df['Age Bracket'] != 'death', 'Population'].sum()
        #df.to_csv('prediction/prediction_2009.csv', index=False)
        return df, sum
    elif(year == 2009):
        data = pd.read_csv('category_overall/2009_group_file.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        sum = df.loc[df['Age Bracket'] != 'death', 'Population'].sum()
        #df.to_csv('prediction/prediction_2011.csv', index=False)
        return df, sum
    elif(year == 2011):
        data = pd.read_csv('category_overall/2011_group_file.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        sum = df.loc[df['Age Bracket'] != 'death', 'Population'].sum()
        #df.to_csv('prediction/prediction_2013.csv', index=False)
        return df, sum
    elif(year == 2013):
        data = pd.read_csv('category_overall/2013_group_file.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        sum = df.loc[df['Age Bracket'] != 'death', 'Population'].sum()
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, sum
    elif(year == 2015):
        data = pd.read_csv('category_overall/2015_group_file.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        sum = df.loc[df['Age Bracket'] != 'death', 'Population'].sum()
        #df.to_csv('prediction/prediction_2017.csv', index=False)
        return df, sum
    else:
        print("dataset not present")

model, t_pop =  model(option)
fig = px.bar(model, x='Age Bracket', y='Population', color = 'Age Bracket' )
predicted_year = option + 2 
st.write("The prediction is for the Year " + str(predicted_year), fig)
st.write("The Total Population for the Year " + str(predicted_year)+ " is ", int(t_pop))

#graph 3 
st.subheader('Overall prediction of Province By Our Model')

#code for province
def model_province(year, province):
    if (province == "Balochistan"):
        data = pd.read_csv('Balochistan/group/group_Balochistan_'+ str(year) +'.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('Balochistan/final_prob_matrix_Balochistan.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2009.csv', index=False)
        return df, int(pred_matrix.sum())
    elif(province == "NWFP"):
        data = pd.read_csv('NWFP/group/group_NWFP_'+ str(year) +'.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('NWFP/final_prob_matrix_NWFP.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2011.csv', index=False)
        return df, int(pred_matrix.sum())
    elif(province == "Sindh"):
        data = pd.read_csv('Sindh/group/group_Sindh_'+ str(year) +'.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('Sindh/final_prob_matrix_Sindh.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2013.csv', index=False)
        return df, int(pred_matrix.sum())
    elif(province == "Punjab"):
        data = pd.read_csv('Punjab/group/group_Punjab_'+ str(year) +'.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('Punjab/final_prob_matrix_Punjab.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, int(pred_matrix.sum())
    else:
        print("dataset not present")


model, t_pop_pro =  model_province(option, option1)
fig = px.bar(model, x='Age Bracket', y='Population', color = 'Age Bracket')
predicted_year = option + 2
st.write("The prediction is for Province "+ option1 + " for the Year " + str(predicted_year), fig)
st.write("The prediction is for Province "+ option1 + " for the Year " + str(predicted_year), t_pop_pro)