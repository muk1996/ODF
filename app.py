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
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 
import os
import time
from load_css import local_css
# from utils import img_to_bytes

# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded

# header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
#     img_to_bytes("header.png")
# )
# st.markdown(
#     header_html, unsafe_allow_html=True,
# )

st.image("header2.png", use_column_width=True)



# t = "<div><span class='highlight red'><span class='bold'>Open Data Forecasting Shown is the predicted population of Pakistan using the census data </span></span></div>"

# st.markdown(t, unsafe_allow_html=True)

# st.write("""
# # Open Data Forecasting
# Shown is the **predicted population** of ***Pakistan*** using the census data
# """)

# bgcolor = st.beta_color_picker("Pick a background color")
# fontcolor = st.beta_color_picker("Pick a font color")

# html_temp = """
# <div style="background-color:{};padding:10px"> 
# <h1 style="color:{};text-align:center">Open Data Forecasting</h1>
# </div>

# """

# st.markdown(html_temp.format(bgcolor, fontcolor),unsafe_allow_html=True)

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# def remote_css(url):
#     st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

# def icon(icon_name):
#     st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

# local_css("style.css")
# remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


 

st.sidebar.markdown("# Navigation Options")
#st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

nav = st.sidebar.selectbox('', ('Homepage', 'Analysis','National Level Prediction', 'Provincial Level Prediction', 'District Level Prediction'))

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
data_country_model = pd.read_csv("country_model.csv")
data_pct_national = pd.read_csv("pak_pct_change.csv")
data_pct_province = pd.read_csv("prov_pct_change.csv")
data_pct_district = pd.read_csv("dist_pct_change.csv")


# def file_selector(folder_path='.'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)

# filename = file_selector()
# st.write('You selected `%s`' % filename)

# # st.write(data_information('sec_b.csv'))

if nav == "Homepage":
    st.write("")
    st.write("### **Overview** ")
    st.image("homepage8.png", use_column_width=True)
    st.write("")
    st.write("### **Benefits of Population Prediction** ")
    st.image("homepage2.png", use_column_width=True)
    st.image("homepage3.png", use_column_width=True)
    st.image("homepage4.png", use_column_width=True)
    st.image("homepage5.png", use_column_width=True)
    st.write("")
    st.write("### **Facts** ")
    st.image("homepage9.png", use_column_width=True)
   
    
elif nav == "Analysis":
    def main():
        activities = ["EDA","Model Building","Plots"]	
        choice = st.sidebar.selectbox("Select Activities",activities)

        if choice == 'EDA':
            st.subheader("Exploratory Data Analysis")

            data = st.file_uploader("Upload a Dataset", type=["csv"])
            if data is not None:
                df = pd.read_csv(data)
                st.dataframe(df.head())

                if st.checkbox("Show Shape"):
                    st.write(df.shape)

                if st.checkbox("Show Columns"):
                    all_columns = df.columns.to_list()
                    st.write(all_columns)

                if st.checkbox("Summary"):
                    st.write(df.describe())

                if st.checkbox("Show Selected Columns"):
                    selected_columns = st.multiselect("Select Columns",all_columns)
                    new_df = df[selected_columns]
                    st.dataframe(new_df)

                if st.checkbox("Show Value Counts"):
                    st.write(df.iloc[:,-1].value_counts())

                if st.checkbox("Correlation Plot(Matplotlib)"):
                    plt.matshow(df.corr())
                    st.pyplot()

                if st.checkbox("Correlation Plot(Seaborn)"):
                    st.write(sns.heatmap(df.corr(),annot=True))
                    st.pyplot()


                if st.checkbox("Pie Plot"):
                    all_columns = df.columns.to_list()
                    column_to_plot = st.selectbox("Select 1 Column",all_columns)
                    pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                    st.write(pie_plot)
                    st.pyplot()



        elif choice == 'Plots':
            st.subheader("Data Visualization")
            data = st.file_uploader("Upload a Dataset", type=["csv"])
            if data is not None:
                df = pd.read_csv(data)
                st.dataframe(df.head())


                if st.checkbox("Show Value Counts"):
                    st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
                    st.pyplot()
            
                # Customizable Plot

                all_columns_names = df.columns.tolist()
                type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
                selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

                if st.button("Generate Plot"):
                    st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

                    # Plot By Streamlit
                    if type_of_plot == 'area':
                        cust_data = df[selected_columns_names]
                        st.area_chart(cust_data)

                    elif type_of_plot == 'bar':
                        cust_data = df[selected_columns_names]
                        st.bar_chart(cust_data)

                    elif type_of_plot == 'line':
                        cust_data = df[selected_columns_names]
                        st.line_chart(cust_data)

                    # Custom Plot 
                    elif type_of_plot:
                        cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
                        st.write(cust_plot)
                        st.pyplot()

        elif choice == 'Model Building':
            st.subheader("Your Own Predictions")
            df_model = st.file_uploader("Upload a Dataset", type=["csv"])
            if df_model is not None:      
                
                
                level = st.selectbox('Select the Level of Prediction',('National', 'Provincial'))
                if level == 'National':

                    def your_model():
                        data = pd.read_csv(df_model)
                        cut_labels_5 = ['children', 'teen', 'young_adult', 'adult', 'senior']
                        cut_bins = [-1, 12, 19, 30, 65, 99]
                        data['age_bin'] = pd.cut(data['age'], bins=cut_bins, labels=cut_labels_5)
                        data_sum = data.groupby(['age_bin'])['weight'].sum().reset_index()
                        matrix = data_sum.to_numpy()
                        data_matrix = matrix.transpose()
                        df = pd.DataFrame(data=data_matrix)
                        new_header = df.iloc[0] #grab the first row for the header
                        df = df[1:] #take the data less the header row
                        df.columns = new_header #set the header row as the df header
                        df = df[['children', 'teen', 'young_adult', 'adult', 'senior']]
                        return df

                    def model_pred(year):
                        import pandas as pd
                        if(year == 2016):
                            df = pd.DataFrame(data = your_model())
                            data = df
                            pop_matrix = data.to_numpy()
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            sum = df.loc[df['Age Bracket'] != 'death', 'Population'].sum()
                            #df.to_csv('prediction/prediction_2017.csv', index=False)
                            return df, sum
                        elif(year == 2017):
                            data = model_pred(2016)[0]
                            pop_matrix = data.transpose()
                            new_header = pop_matrix.iloc[0] #grab the first row for the header
                            pop_matrix = pop_matrix[1:] #take the data less the header row
                            pop_matrix.columns = new_header
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
                            df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            #df.to_csv('prediction/prediction_2015.csv', index=False)
                            return df, df2
                        elif(year == 2018):
                            data = model_pred(2017)[0]
                            pop_matrix = data.transpose()
                            new_header = pop_matrix.iloc[0] #grab the first row for the header
                            pop_matrix = pop_matrix[1:] #take the data less the header row
                            pop_matrix.columns = new_header
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
                            df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            #df.to_csv('prediction/prediction_2015.csv', index=False)
                            return df, df2
                        elif(year == 2019):
                            data = model_pred(2018)[0]
                            pop_matrix = data.transpose()
                            new_header = pop_matrix.iloc[0] #grab the first row for the header
                            pop_matrix = pop_matrix[1:] #take the data less the header row
                            pop_matrix.columns = new_header
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
                            df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            #df.to_csv('prediction/prediction_2015.csv', index=False)
                            return df, df2
                        elif(year == 2020):
                            data = model_pred(2019)[0]
                            pop_matrix = data.transpose()
                            new_header = pop_matrix.iloc[0] #grab the first row for the header
                            pop_matrix = pop_matrix[1:] #take the data less the header row
                            pop_matrix.columns = new_header
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
                            df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            #df.to_csv('prediction/prediction_2015.csv', index=False)
                            return df, df2
                        elif(year == 2021):
                            data = model_pred(2020)[0]
                            pop_matrix = data.transpose()
                            new_header = pop_matrix.iloc[0] #grab the first row for the header
                            pop_matrix = pop_matrix[1:] #take the data less the header row
                            pop_matrix.columns = new_header
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
                            df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            #df.to_csv('prediction/prediction_2015.csv', index=False)
                            return df, df2
                        elif(year == 2022):
                            data = model_pred(2021)[0]
                            pop_matrix = data.transpose()
                            new_header = pop_matrix.iloc[0] #grab the first row for the header
                            pop_matrix = pop_matrix[1:] #take the data less the header row
                            pop_matrix.columns = new_header
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
                            df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            #df.to_csv('prediction/prediction_2015.csv', index=False)
                            return df, df2
                        elif(year == 2023):
                            data = model_pred(2022)[0]
                            pop_matrix = data.transpose()
                            new_header = pop_matrix.iloc[0] #grab the first row for the header
                            pop_matrix = pop_matrix[1:] #take the data less the header row
                            pop_matrix.columns = new_header
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
                            df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            #df.to_csv('prediction/prediction_2015.csv', index=False)
                            return df, df2
                        elif(year == 2024):
                            data = model_pred(2023)[0]
                            pop_matrix = data.transpose()
                            new_header = pop_matrix.iloc[0] #grab the first row for the header
                            pop_matrix = pop_matrix[1:] #take the data less the header row
                            pop_matrix.columns = new_header
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
                            df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            #df.to_csv('prediction/prediction_2015.csv', index=False)
                            return df, df2
                        elif(year == 2025):
                            data = model_pred(2024)[0]
                            pop_matrix = data.transpose()
                            new_header = pop_matrix.iloc[0] #grab the first row for the header
                            pop_matrix = pop_matrix[1:] #take the data less the header row
                            pop_matrix.columns = new_header
                            matrix = pd.read_csv('own_model.csv')
                            prob_matrix = matrix.to_numpy()
                            pred_matrix = pop_matrix.dot(prob_matrix)
                            df = pd.DataFrame(data = pred_matrix.transpose(), columns=["Population"])
                            df.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            df2 = pd.DataFrame(data = pred_matrix.sum(), columns=["Total"])
                            df2.insert(0, "Age Bracket", ["children", "teen", "young_adult", "adult", "senior"], True)
                            #df.to_csv('prediction/prediction_2015.csv', index=False)
                            return df, df2
                    year = st.selectbox('Please Select the Year you want prediction for',
                                        (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025))

                    model_predict, t_pop_predict =  model_pred(year)
                    st.markdown("### Bar Chart")
                    fig = px.bar(model_predict, x='Age Bracket', y='Population', color = 'Age Bracket' ) 
                    st.write("The **Prediction** is for the Year " + str(year), fig)
                    st.write("The **Total Population** for the Year " + str(year)+ " is ", int(t_pop_predict.loc[0:4][["Total"]].sum()))
                    #st.write("**Change in Literacy** in the Year "+ str(year)+" compared to " + str(year-1)+ " is ", int(t_pop.loc[0:4][["Total"]].sum()*data_pct_national["pct_change"].sum()))
                    st.markdown("### Pie Chart")
                    fig = px.pie(model_predict, values='Population', names='Age Bracket')
                    st.write("The Pie Chart of the Total Population for the Year " + str(year), fig)
                    st.write("The **Total Population** for the Year " + str(year)+ " is ", int(t_pop_predict.loc[0:4][["Total"]].sum()))
                    #st.write("The **Change in Literacy** in the Year "+ str(year)+" compared to " + str(year-1)+ " is ", int(t_pop.loc[0:4][["Total"]].sum()*data_pct_national["pct_change"].sum()))
                    st.markdown("### Map")
                    fig = px.scatter_mapbox(model_predict, lat=data_country_model['lat'], lon=data_country_model['lon'], 
                                size=t_pop_predict["Total"], color='Age Bracket', color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=5,
                                mapbox_style="carto-positron")
                    st.write("The Map for the Year " + str(year)) 
                    st.plotly_chart(fig)
                
                else:    
                    year = st.selectbox('Please Select the Year you want prediction for',
                                        (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025))

                    province = st.selectbox('Please Select the Province you want prediction for',
                                            ("Sindh", "Balochistan", "Punjab", "NWFP"))

                    #graph 3 
                    st.markdown('## **Provincial Level Prediction Using HMM Model** ')

                    def your_model_pro(province):
                        data = pd.read_csv(df_model)
                        cut_labels_5 = ['children', 'teen', 'young_adult', 'adult', 'senior']
                        cut_bins = [-1, 12, 19, 30, 65, 99]
                        data['age_bin'] = pd.cut(data['age'], bins=cut_bins, labels=cut_labels_5)
                        data = data.loc[data['province'] == province]
                        data_sum = data.groupby(['age_bin'])['weight'].sum().reset_index()
                        matrix = data_sum.to_numpy()
                        data_matrix = matrix.transpose()
                        df = pd.DataFrame(data=data_matrix)
                        new_header = df.iloc[0] #grab the first row for the header
                        df = df[1:] #take the data less the header row
                        df.columns = new_header #set the header row as the df header
                        df = df[['children', 'teen', 'young_adult', 'adult', 'senior']]
                        return df
                    #code for province
                    def model_predict(year, province):
                        if(year == 2016):
                            import os
                            df = pd.DataFrame(data = your_model_pro(province))
                            data = df
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
                            data = model_predict(2016, province)[0]
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
                            data = model_predict(2017, province)[0]
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
                            data = model_predict(2018, province)[0]
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
                            data = model_predict(2019, province)[0]
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
                            data = model_predict(2020, province)[0]
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
                            data = model_predict(2021, province)[0]
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
                            data = model_predict(2022, province)[0]
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
                            data = model_predict(2023, province)[0]
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
                            data = model_predict(2024, province)[0]
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


                    model_predict_prov, t_pop__pred_pro =  model_predict(year, province)
                    st.markdown("### Bar Chart")
                    fig = px.bar(model_predict_prov, x='Age Bracket', y='Population', color = 'Age Bracket')
                    st.write("The prediction is for Province "+ province + " for the Year " + str(year), fig)
                    st.write("The prediction is for Province "+ province + " for the Year " + str(year), int(t_pop__pred_pro.Total.sum()))
                    # st.write("The Change in Literate people in the Year "+ str(year)+" compared to " + str(year-1)+ " is ", int(t_pop_pro.loc[0:4][["Total"]].sum()*data_pct_province.loc[data_pct_province["province"] == province, "pct_change"].sum()))
                    st.markdown("### Pie Chart")
                    fig = px.pie(model_predict_prov, values='Population', names='Age Bracket')
                    st.write("The Pie Chart of the "+ province + " for the Year " + str(year), fig)
                    st.write("The prediction is for Province "+ province + " for the Year " + str(year), int(t_pop__pred_pro.Total.sum()))
                    # st.write("The Change in Literate people in the Year "+ str(year)+" compared to " + str(year-1)+ " is ", int(t_pop_pro.loc[0:4][["Total"]].sum()*data_pct_province.loc[data_pct_province["province"] == province, "pct_change"].sum()))
                    st.markdown("### Map")
                    
                    fig = px.scatter_mapbox(model_predict_prov, lat=data_pro.loc[data_pro['province'] == province, 'lat'], lon=data_pro.loc[data_pro['province'] == province, 'lon'], 
                                size=t_pop__pred_pro["Total"], color='Age Bracket', color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=6,
                                mapbox_style="carto-positron")
                    st.write("The Map of "+ province + " for the Year " + str(year))           
                    st.plotly_chart(fig)
    if __name__ == '__main__':
        main()

elif nav == "National Level Prediction":
    # st.sidebar.title("Predicting the Population")
    # st.sidebar.markdown("Using the census data of 🇵🇰 - we predict the population of the country in the coming years  ")
    # st.sidebar.markdown("### Predicted population per year")

    pred = st.sidebar.selectbox('Please Select the Type of Prediction', ("Population Prediction", "Literacy Prediction"))
    
    if pred == "Population Prediction":

        year = st.sidebar.selectbox(
                'Please Select the Year you want prediction for',
                (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025))

        #paragraph 
        st.markdown('## **National Level Prediction** ')

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
        st.markdown("### Bar Chart")
        fig = px.bar(model_pak, x='Age Bracket', y='Population', color = 'Age Bracket' ) 
        st.write("The **Prediction** is for the Year " + str(year), fig)
        st.write("The **Total Population** for the Year " + str(year)+ " is ", int(t_pop.loc[0:4][["Total"]].sum()))
        
        st.markdown("### Pie Chart")
        fig = px.pie(model_pak, values='Population', names='Age Bracket')
        st.write("The Pie Chart of the Total Population for the Year " + str(year), fig)
        st.write("The **Total Population** for the Year " + str(year)+ " is ", int(t_pop.loc[0:4][["Total"]].sum()))
        
        st.markdown("### Map")
        fig = px.scatter_mapbox(model_pak, lat=data_country['lat'], lon=data_country['lon'], 
                    size=t_pop["Total"], color='Age Bracket', color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=5,
                    mapbox_style="carto-positron")

        st.write("The Map for the Year " + str(year)) 
        st.plotly_chart(fig)
    
    else:
        year = st.sidebar.selectbox(
            'Please Select the Year you want prediction for',
            (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025))

        #paragraph 
        st.markdown('## **National Level Prediction** ')

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
        lit = int(t_pop.loc[0:4][["Total"]].sum()*data_pct_national["pct_change"].sum())
        lit_pop = int(t_pop.loc[0:4][["Total"]].sum()*data_pct_national["literate"].sum() + lit)
        #fig = plt.plot('year', 'lit_pop', 'ro', label='Point A'); plt.legend()
        fig = go.Figure(go.Indicator(
            mode = "number+gauge",
            value = lit_pop,
            title = {'text': "Literacy Prediction", 'font': {'color': "grey", 'size': 24}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'bar': {'color': "#026212"}, 'bgcolor': "#ebebeb"},
            number = {'font': {'color': "#304233"}}
        ))

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(fig)
        st.write(" ")
        st.write("The Population of Literate People in the Year " + str(year) + " is ", lit_pop)


        #st.write("The **Change in Literacy** in the Year "+ str(year)+" compared to " + str(year-1)+ " is ", int(t_pop.loc[0:4][["Total"]].sum()*data_pct_national["pct_change"].sum()))

elif nav == "Provincial Level Prediction":
    # st.sidebar.title("Predicting the Population")
    # st.sidebar.markdown("Using the census data of 🇵🇰 - we predict the population of the country in the coming years  ")
    # st.sidebar.markdown("### Predicted population per year")

    pred = st.sidebar.selectbox('Please Select the Type of Prediction', ("Population Prediction", "Literacy Prediction"))
    
    if pred == "Population Prediction":

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

        #graph 3 
        st.markdown('## **Provincial Level Prediction** ')

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
        st.markdown("### Bar Chart")
        fig = px.bar(model_prov, x='Age Bracket', y='Population', color = 'Age Bracket')
        st.write("The prediction is for Province "+ province + " for the Year " + str(year), fig)
        st.write("The prediction is for Province "+ province + " for the Year " + str(year), int(t_pop_pro.Total.sum()))
        st.markdown("### Pie Chart")
        fig = px.pie(model_prov, values='Population', names='Age Bracket')
        st.write("The Pie Chart of the "+ province + " for the Year " + str(year), fig)
        st.write("The prediction is for Province "+ province + " for the Year " + str(year), int(t_pop_pro.Total.sum()))
        
        st.markdown("### Map")
        
        fig = px.scatter_mapbox(model_prov, lat=data_pro.loc[data_pro['province'] == province, 'lat'], lon=data_pro.loc[data_pro['province'] == province, 'lon'], 
                    size=t_pop_pro["Total"], color='Age Bracket', color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=6,
                    mapbox_style="carto-positron")
        st.write("The Map of "+ province + " for the Year " + str(year))           
        st.plotly_chart(fig)
    
    else:
        year = st.sidebar.selectbox(
            'Please Select the Year you want prediction for',
            (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025))

        province = st.sidebar.selectbox(
            'Please Select the Province you want prediction for',
            ("Sindh", "Balochistan", "Punjab", "NWFP"))


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
        lit = int(t_pop_pro.loc[0:4][["Total"]].sum()*data_pct_province.loc[data_pct_province["province"] == province, "pct_change"].sum())
        lit_pop = int(t_pop_pro.loc[0:4][["Total"]].sum()*data_pct_province.loc[data_pct_province["province"] == province, "literate"].sum() + lit)
        #fig = plt.plot('year', 'lit_pop', 'ro', label='Point A'); plt.legend()
        fig = go.Figure(go.Indicator(
            mode = "number+gauge",
            value = lit_pop,
            title = {'text': "Literacy Prediction", 'font': {'color': "grey", 'size': 24}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'bar': {'color': "#026212"}, 'bgcolor': "#ebebeb"},
            number = {'font': {'color': "#304233"}}
        ))

        #st.write("The Population of Literate People in the Year " + str(year) + " is ", lit_pop, fig)
        #st.write("The Change in Literate people in the Year "+ str(year)+" compared to " + str(year-1)+ " is ", int(t_pop_pro.loc[0:4][["Total"]].sum()*data_pct_province.loc[data_pct_province["province"] == province, "pct_change"].sum()))
        #fig = px.bar(model_prov, x='Age Bracket', y='Population', color = 'Age Bracket')
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(fig)
        st.write(" ")
        st.write("The Population of Literate People in the Year " + str(year) + " is ", lit_pop)

        


elif nav == "District Level Prediction":
    #sidebar
    # st.sidebar.title("Predicting the Population")
    # st.sidebar.markdown("Using the census data of 🇵🇰 - we predict the population of the country in the coming years  ")
    # st.sidebar.markdown("### Predicted population per year")

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
    #graph 4
    st.markdown('## **District Level Prediction  Model** ')

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
    st.markdown("### Bar Chart")
    fig = px.bar(model_dist, x='Age Bracket', y='Population', color = 'Age Bracket')
    st.write("The prediction is for District "+ district + " for the Year " + str(year), fig)
    st.write("The prediction is for District "+ district + " for the Year " + str(year), int(t_pop_dist.Total.sum()))
    st.markdown("### Pie Chart")
    fig = px.pie(model_dist, values='Population', names='Age Bracket')
    st.write("The Pie Chart of the "+ district + " for the Year " + str(year), fig)
    st.write("The prediction is for District "+ district + " for the Year " + str(year), int(t_pop_dist.Total.sum()))
    #st.write("The Change in Literate people in the Year "+ str(year)+" compared to " + str(year-1)+ " is ", int(t_pop_dist.loc[0:4][["Total"]].sum()*data_pct_district.loc[data_pct_district["district"] == district, "pct_change"].sum()))
    st.markdown("### Map")
    fig = px.scatter_mapbox(model_dist, lat=df.loc[df['district'] == district, 'lat'], lon=df.loc[df['district'] == district, 'lon'], 
                size=t_pop_dist["Total"], color='Age Bracket', color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=6,
                mapbox_style="carto-positron")
    st.plotly_chart(fig)

    # def data_information(dataset):
    #     data = pd.read_csv(dataset)
    #     data_shape = print("The number of rows in the Data are " + str(data.shape[0]) + ".\n" + "The number of columns in the Data are " + str(data.shape[1])+ "."+ "\n")
    #     data_type = print("Following are the data types of each column \n" + str(data.dtypes) + "\n")
    #     data_info_text = print("Following are the data types of each column \n ")
    #     data_info = print(str(data.info())+ "\n")
    #     df_numeric = data.select_dtypes(include=[np.number])
    #     numeric_cols = df_numeric.columns.values
    #     numeric_columns_text = print("The list of numeric columns in the dataset are")
    #     numeric_columns_values = print('\n'.join(map(str, numeric_cols))+ "\n")
    #     numeric_columns_description = print("The Description of numeric columns in the dataset are \n" + str(df_numeric.describe())+ "\n")
    #     df_non_numeric = data.select_dtypes(exclude=[np.number])
    #     non_numeric_cols = df_non_numeric.columns.values
    #     non_numeric_columns_text = print("The list of non-numeric columns in the dataset are")
    #     non_numeric_columns_values = print('\n'.join(map(str, non_numeric_cols))+ "\n")
    #     missing_values_text = print("The percentage of missing values in each column")
    #     for col in data.columns:
    #         pct_missing = np.mean(data[col].isnull())
    #         missing_values = print('{} - {}%'.format(col, round(pct_missing*100)))
    #     return data_shape, data_type, data_info_text, data_info, numeric_columns_text, numeric_columns_values, numeric_columns_description, non_numeric_columns_text,non_numeric_columns_values, missing_values_text, missing_values 

    # def data_cleaning(dataset):
    #     df_numeric = dataset.select_dtypes(include=[np.number])
    #     numeric_cols = df_numeric.columns.values
    #     for col in numeric_cols:
    #         missing = dataset[col].isnull()
    #         num_missing = np.sum(missing)
    #         if num_missing > 0:  # only do the imputation for the columns that have missing values.
    #             print('imputing missing values for: {}'.format(col))
    #             dataset['{}_ismissing'.format(col)] = missing
    #             med = dataset[col].median()
    #             dataset[col] = dataset[col].fillna(med)
    #     df_non_numeric = dataset.select_dtypes(exclude=[np.number])
    #     non_numeric_cols = df_non_numeric.columns.values
    #     for col in non_numeric_cols:
    #         missing = dataset[col].isnull()
    #         num_missing = np.sum(missing)
    #         if num_missing > 0:  # only do the imputation for the columns that have missing values.
    #             print('imputing missing values for: {}'.format(col))
    #             dataset['{}_ismissing'.format(col)] = missing
    #             top = dataset[col].describe()['top'] # impute with the most frequent value.
    #             dataset[col] = dataset[col].fillna(top)
    #     return dataset

    # def imputing_numeric_data(dataset):
    #     df_numeric = df.select_dtypes(include=[np.number])
    #     numeric_cols = df_numeric.columns.values
    #     for col in numeric_cols:
    #         missing = df[col].isnull()
    #         num_missing = np.sum(missing)
    #         if num_missing > 0:  # only do the imputation for the columns that have missing values.
    #             print('imputing missing values for: {}'.format(col))
    #             df['{}_ismissing'.format(col)] = missing
    #             med = df[col].median()
    #             df[col] = df[col].fillna(med)
        
    # def imputing_non_numeric_data(dataset):
    #     df_non_numeric = df.select_dtypes(exclude=[np.number])
    #     non_numeric_cols = df_non_numeric.columns.values
    #     for col in non_numeric_cols:
    #         missing = df[col].isnull()
    #         num_missing = np.sum(missing)
    #         if num_missing > 0:  # only do the imputation for the columns that have missing values.
    #             print('imputing missing values for: {}'.format(col))
    #             df['{}_ismissing'.format(col)] = missing
    #             top = df[col].describe()['top'] # impute with the most frequent value.
    #             df[col] = df[col].fillna(top)

    # num_rows = len(df.index)
    # low_information_cols = [] #

    # for col in df.columns:
    #     cnts = df[col].value_counts(dropna=False)
    #     top_pct = (cnts/num_rows).iloc[0]
        
    #     if top_pct > 0.95:
    #         low_information_cols.append(col)
    #         print('{0}: {1:.5f}%'.format(col, top_pct*100))
    #         print(cnts)
    #         print()

    # # we know that column 'id' is unique, but what if we drop it?
    # df_dedupped = df.drop('hhcode', axis=1).drop_duplicates()

    # # there were duplicate rows
    # print(df.shape)
    # print(df_dedupped.shape)

    # # make everything lower case.
    # df_non_numeric = df.select_dtypes(exclude=[np.number])
    # non_numeric_cols = df_non_numeric.columns.values

    # for col in non_numeric_cols:
    #     df[col+'lower'] = df[col].str.lower()
    #     df[col+'lower'].value_counts(dropna=False)
