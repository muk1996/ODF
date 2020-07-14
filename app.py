import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import altair as alt
from datetime import datetime
from pandas import Series
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.title("Open Data Forecasting")

#st.image("image_3.jpg", use_column_width=True)

#sidebar
st.sidebar.title("Predicting the Population")

#description on the mainpage
st.markdown("Using the census data of 🇵🇰 - we predict the future population of the country ")

#copying it to the sidebar
st.sidebar.markdown("Using the census data of 🇵🇰 - we predict the population of the country in the coming years  ")

#loading the csv file
#data_url = ("C:/Users/hp/Desktop/Python dashboards/streamlit/combined_districts.csv")

DATA_URL = ('combined_districts.csv')

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    #data['Year'] = pd.to_datetime(data['Year']).dt.strftime('%Y')
    return data
df = load_data()
data_pro = pd.read_csv('province.csv')
data_country = pd.read_csv("country.csv")

st.sidebar.markdown("### Predicted population per year")
st.markdown("### Actual Population based on our Dataset")
sum = df.groupby(['year'])['Total'].sum().reset_index()

#histogram or piechart
select = st.sidebar.selectbox('Visualization Type', ['Histogram', 'Pie Chart',"Map"], key='1')
#if not st.sidebar.checkbox():
    #st.markdown("## Population Predicition")
if select == "Histogram":
    fig = px.bar(sum, x='year', y='Total', color='year')
    st.plotly_chart(fig)
elif select == "Pie Chart":
    fig = px.pie(sum, values='Total', names='year')
    st.plotly_chart(fig)
else:
    fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="district", size="Total",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=3,
                  mapbox_style="carto-positron")
    st.plotly_chart(fig)


year = st.sidebar.selectbox(
        'Please Select the Year you want prediction for',
        (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025))

province = st.sidebar.selectbox(
        'Please Select the Province you want prediction for',
        ("Sindh", "Balochistan", "Punjab", "NWFP"))

def select_province(province):
    
    if (province == "Sindh"):
        return ('Badin', 'Dadu', 'Ghotki', 'Hyderabad', 'Jacobabad', 'Jamshoro', 'Karachi',
            'Kashmore', 'Khairpur', 'Larkana', 'Matiari', 'Mirpurkhas', 'Naushahro Feroze', 
            'Nawabshah', 'Sanghar', 'Shahdadkot', 'Shikarpur', 'Sukkur', 'Tando Allah Yar', 'Tando Mohammad Khan',
            'Tharparkar', 'Thatta','Umer kot')
    elif(province == "Punjab"):
        return ('Attock','Bahawalnagar','Bahawalpur', 'Bhakkar', 'Chakwal', 'D.G.Khan',
            'Faisalabad','Gujranwala','Gujrat', 'Hafizabad', 'Islamabad', 'Jehlum', 'Jhang', 'Kasur', 
            'Khanewal', 'Khushab', 'Lahore', 'Layyah', 'Lodhran', 'Mandi Bahauddin','Mianwali', 
            'Multan', 'Muzaffargarh', 'Nankana Sahib', 'Narowal', 'Okara', 'Pakpattan', 'Rahim Yar Khan', 
            'Rajanpur', 'Rawalpindi', 'Sahiwal', 'Sarghodha', 'Sheikhupura', 'Sialkot', 'T.T.Singh', 'Vehari')
    elif(province == "NWFP"):
        return ('Abbottabad',  'Bannu', 'Batagram', 'Bonair', 'Charsada',
            'Chitral', 'D.I.Khan', 'Hangu', 'Haripur', 'Karak', 'Kohat', 'Kohistan',
            'Lakki Marwat', 'Lower Dir', 'Malakand', 'Mansehra', 'Mardan', 'Nowshera', 'Peshawar',
            'Shangla',  'Swabi', 'Swat', 'Tank', 'Upper Dir')
    elif(province == "Balochistan"):
        return ('Awaran', 'Barkhan', 'BolanKachhi', 'Chagai', 'Dera_Bughti', 'Gwadar', 'Hangu', 
            'Jaffarabad', 'Jhal Magsi', 'Kalat','Khuzdar', 'Kohlu', 'Lasbela',  'Loralai',
            'Mastung', 'Musa Khel',  'Nasirabad', 'Pishin','Qilla Abdullah', 
            'Qilla Saifullah', 'Quetta', 'Sheerani', 'Sibbi', 'Washuk', 'Ziarat')


district = st.sidebar.selectbox(
        'Please select the District you want prediction for', (select_province(province)))
        
#paragraph 
st.subheader('Overall prediction of Pakistan By Our Model')

#code for overall
def model(year):
    if(year == 2016):
        data = pd.read_csv('category_pakistan/2015_group_file.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        sum = df.loc[df['Age Bracket'] != 'death', 'Population'].sum()
        #df.to_csv('prediction/prediction_2017.csv', index=False)
        return df, sum
    elif(year == 2017):
        data = model(2016)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2018):
        data = model(2017)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2019):
        data = model(2018)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2020):
        data = model(2019)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2021):
        data = model(2020)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2022):
        data = model(2021)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2023):
        data = model(2022)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2024):
        data = model(2023)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2025):
        data = model(2024)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv('F_L_M.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior", "death"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2

model_pak, t_pop =  model(year)
if (select == "Histogram"):
    fig = px.bar(model_pak, x='Age Bracket', y='Population', color = 'Age Bracket' ) 
    st.write("The prediction is for the Year " + str(year), fig)
    st.write("The Total Population for the Year " + str(year)+ " is ", int(t_pop.loc[0:4][["Total"]].sum()))
elif (select == "Pie Chart"):
    fig = px.pie(model_pak, values='Population', names='Age Bracket')
    st.write("The Pie Chart of the Total Population for the Year " + str(year), fig)
    st.write("The Total Population for the Year " + str(year)+ " is ", int(t_pop.loc[0:4][["Total"]].sum()))
else:
    fig = px.scatter_mapbox(model_pak, lat=data_country['lat'], lon=data_country['lon'], 
                size=t_pop["Total"], color='Age Bracket', color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=5,
                mapbox_style="carto-positron")
    st.plotly_chart(fig)


#graph 3 
st.subheader('Overall prediction of Province By Our Model')

#code for province
def model(year, province):
    if(year == 2016):
        import os
        data = pd.read_csv(province+'/group/group_'+province+'_2015.csv')
        pop_matrix = data.to_numpy()
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df["Population"].sum()
    elif(year == 2017):
        import os
        data = model(2016, province)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2018):
        import os
        data = model(2017, province)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2019):
        import os
        data = model(2018, province)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2020):
        import os
        data = model(2019, province)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2021):
        import os
        data = model(2020, province)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2022):
        import os
        data = model(2021, province)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2023):
        import os
        data = model(2022, province)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2024):
        import os
        data = model(2023, province)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2025):
        import os
        data = model(2024, province)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        matrix = pd.read_csv(province+'/final_prob_matrix_'+province+'.csv')
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2


model_prov, t_pop_pro =  model(year, province)
if (select == "Histogram"):
    fig = px.bar(model_prov, x='Age Bracket', y='Population', color = 'Age Bracket')
    st.write("The prediction is for Province "+ province + " for the Year " + str(year), fig)
    st.write("The prediction is for Province "+ province + " for the Year " + str(year), int(t_pop_pro.Total.sum()))
elif (select == "Pie Chart"):
    fig = px.pie(model_prov, values='Population', names='Age Bracket')
    st.write("The Pie Chart of the "+ province + " for the Year " + str(year), fig)
    st.write("The prediction is for Province "+ province + " for the Year " + str(year), int(t_pop_pro.Total.sum()))
else:
    fig = px.scatter_mapbox(model_prov, lat=data_pro.loc[data_pro['province'] == province, 'lat'], lon=data_pro.loc[data_pro['province'] == province, 'lon'], 
                size=t_pop_pro["Total"], color='Age Bracket', color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=6,
                mapbox_style="carto-positron")
    st.plotly_chart(fig)

#graph 4
st.subheader('Overall prediction of district By Our Model')

def model(year, district):
    if(year == 2016):
        import os
        path = "group_district/group_"+district+"_2015.csv"
        data = pd.read_csv(path)
        pop_matrix = data.to_numpy()
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df["Population"].sum()
    elif(year == 2017):
        import os
        data = model(2016, district)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2018):
        import os
        data = model(2017, district)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2019):
        import os
        data = model(2018, district)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2020):
        import os
        data = model(2019, district)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2021):
        import os
        data = model(2020, district)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2022):
        import os
        data = model(2021, district)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2023):
        import os
        data = model(2022, district)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2024):
        import os
        data = model(2023, district)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2
    elif(year == 2025):
        import os
        data = model(2024, district)[0]
        pop_matrix = data.transpose()
        new_header = pop_matrix.iloc[0] #grab the first row for the header
        pop_matrix = pop_matrix[1:] #take the data less the header row
        pop_matrix.columns = new_header
        path1 = "matrix_2015/new_matrix_"+district+"_2015.csv"
        matrix = pd.read_csv(path1)
        prob_matrix = matrix.to_numpy()
        pred_matrix = pop_matrix.dot(prob_matrix)
        df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
        df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
        df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
        #df.to_csv('prediction/prediction_2015.csv', index=False)
        return df, df2

model_dist, t_pop_dist =  model(year, district)
if (select == "Histogram"):
    fig = px.bar(model_dist, x='Age Bracket', y='Population', color = 'Age Bracket')
    st.write("The prediction is for District "+ district + " for the Year " + str(year), fig)
    st.write("The prediction is for District "+ district + " for the Year " + str(year), int(t_pop_dist.Total.sum()))
elif(select =="Pie Chart"):
    fig = px.pie(model_dist, values='Population', names='Age Bracket')
    st.write("The Pie Chart of the "+ district + " for the Year " + str(year), fig)
    st.write("The prediction is for District "+ district + " for the Year " + str(year), int(t_pop_dist.Total.sum()))
else:
    fig = px.scatter_mapbox(model_dist, lat=df.loc[df['district'] == district, 'lat'], lon=df.loc[df['district'] == district, 'lon'], 
                size=t_pop_dist["Total"], color='Age Bracket', color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=6,
                mapbox_style="carto-positron")
    st.plotly_chart(fig)