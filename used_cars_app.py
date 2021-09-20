import pandas as pd
import numpy as np

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from PIL import Image

import streamlit as st

st.title("Used Vehicles on Craigslist")

mapbox_access_token= "pk.eyJ1Ijoic2FyYS1iaGFtZGFuIiwiYSI6ImNrdGJ4eGVzNTIwdTUybnBkeXVibzI4ODQifQ.l34h5B7ThJQc-nRlHB-eHg"

@st.cache
def load_data():
    df = pd.read_csv('vehicles_cleaned.csv')
    return df

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load data into the dataframe.
df = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.sidebar.markdown("## Side Panel")
st.sidebar.markdown("Use this panel to explore the dataset and create own viz.")
data_load_state.text('')
st.sidebar.subheader('Quick  Explore')

# Basic Info
if st.sidebar.checkbox("Basic Info"):
    img=Image.open('cars.jfif')
    st.image(img,width=600)

    st.markdown("""
        Craigslist is the world's largest collection of used vehicles for sale. This data is scraped every few months, 
        it contains most all relevant information that Craigslist provides on car sales including columns like price, condition, 
        manufacturer, latitude/longitude, and 18 other categories.\n 
        The dataset specifically includes used vehicles posted for sale on Craigslist during April 2021 and can be found on Kaggle:
        https://www.kaggle.com/austinreese/craigslist-carstrucks-data/version/10 
        """)

    # Visualize vehicles conditions
    #st.subheader("Condition of Used Vehicles for Sale on Craigslist")
    condition_pie=px.pie(df, "condition", labels=dict(condition="Condition"), title=("Condition of Used Vehicles for Sale on Craigslist"))
    st.write(condition_pie)

    st.markdown("""
        With the numerous options and wide price ranges for used vehicles on Craigslist, it is difficult for a user to find the vehicles they need on one hand 
        and set a price for a used vehicles they want to sell on the other hand.\n
        This app allows the user to explore the used vehicles available on Craigslist by location, manufacturer, release year, odometer and other variables through interactive visualizations.\n
        It also depicts the relationship of price with certain variables that can be further developed into a model to predict prices uisng machine learning.
        """)


# Visualize Raw Data
if st.sidebar.checkbox("Dataset Quick Look"):
    st.subheader("Dataset Quick Look")
    st.write(df.head(10))
    st.markdown("For simplicity, some features were removed from the dataset such as urls, description, model and vehicle ID.")

# Visualize Statistical Info
if st.sidebar.checkbox("Statistical Info"):
    st.subheader("Statistical Info on Numerical Features")
    stats= df.describe().reset_index().rename(columns={'index':'statistic'}).round(2)
    table = ff.create_table(stats)
    st.write(table)
    if st.button("Check Categorical Features Too"):
        #st.subheader("Statistical Info on Categorical Features")
        stats= df.describe(exclude='number').T.reset_index().rename(columns={'index':'feature'})
        table2 = ff.create_table(stats)
        st.write(table2)

st.sidebar.subheader('Create Own Visualization')

# Visualize Used Cars Location
if st.sidebar.checkbox("Used Vehicles Locations"):
    st.subheader("Location of Used Vehicles on Craigslist")
    option=st.multiselect("Explore specific states on the map uisng the multiselection tool:", df['state'].unique(), default=df['state'].unique())
    
    if option:
        data = [
        go.Scattermapbox(
            lat=df[df.state.isin(option)].lat,
            lon=df[df.state.isin(option)].long,
            mode='markers',
            marker=dict(
                size=5,
                color= 'rgb(255, 102, 102)',
                opacity=0.55,
            ),
            text= df[df.state.isin(option)].region,
            hoverinfo='text'
        )]

        layout = go.Layout(
            autosize=True,
            hovermode='closest',
            showlegend=False,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0,   #inclination as left or right
                center=dict(     #center of map specified manually in this case
                    lat=38,
                    lon=-94
                ),
                pitch=0,   #inclination downwards as if looking from buttom
                zoom=3,   #map zoom level
                style='light'  #light or dark mode (black map)
            ),
        )

        fig = dict(data=data, layout=layout)
        st.write(fig)
    st.markdown("The mismatch between the actual location and selected state in some places is due to data entry errors by the users when dropping the pin on the map on Craigslist website.")

# Visualize Price vs Manufacturer
if st.sidebar.checkbox("Used Vehicle Prices by Manufacturer"):
    st.subheader("Used Vehicle Prices per Top 10 Manufacturers over Release Years")
    
    #select top 10 manufacturers
    freq_manufacturers= list(df.manufacturer.value_counts().to_frame().head(10).index)
    df3=df[df['manufacturer'].isin(freq_manufacturers)]
    
    years=list(df3.year.unique())
    years.sort()

    release_year= st.slider('Drag the slider to check desired vehicle release year',1992,2022, 2018)
    fig= px.box(df3[df3.year == release_year], x="manufacturer", y="price", color='manufacturer', labels=dict(manufacturer= "Manufacturer", price= 'Price'))
    st.write(fig)
    st.markdown("Prices vary per manufacturer per year. Vehicles produced by RAM had the highest average prices compared to other manufacturers which is correlated with car size as well as they produce large vehicles.")

    if st.button("Check Vehicle Condition of Top 10 Manufacturers"):
        df_stack= df3.groupby(['manufacturer', 'condition']).size().reset_index().rename(columns={0:'Counts'})
        
        fig = px.bar(df_stack, x="manufacturer", y='Counts', color='condition', barmode='stack')
        fig.update_layout(title="Vehicles Condition Across Top 10 Manufacturers", xaxis_title="Manufacturer") 
        st.write(fig)

# Visualize relationship of price with odometer and year
if st.sidebar.checkbox("Correlation Analysis of Used Vehicle Prices"):
    st.subheader("Relationship of Price with Release Year & Odometer")
    fig= px.scatter_matrix(df, dimensions=["price", "year", "odometer"], color="size", opacity=0.16, 
        labels=dict(price= "Price", odometer="Odometer", year="Release Year", size="Vehicle Size"))
    st.write(fig)
    st.markdown('As expected, a positive relationship is noted between release year and used vehicle price versus a negative relationship between odometer and used vehicle price. This can be developed further into a machine learning model for price prediction.')
    st.markdown("For more on price vs odometer and release year, select the last 2 checkboxes in the side panel.")
    if st.sidebar.checkbox("Price vs Odometer by Vehicle Type"):
        freq_types=list(df['type'].value_counts().to_frame().head(3).index)
        df2= df[df['type'].isin(freq_types)]
        fig= px.scatter(df2[:3000], x= "odometer", y="price", color= "drive", size= "year", size_max=3.5, hover_name="manufacturer", facet_col= "type", 
            labels=dict(price= "Price", odometer="Odometer", drive="Drive", type="Type"))
        st.subheader("Price vs Odometer by Vehicle Type")
        st.write(fig)
            
    if st.sidebar.checkbox("Prices vs Odometer over Vehicle Release Years"):
        years=list(df.year.unique())
        years.sort()
        fig= px.scatter(df, x="odometer", y="price", color= "drive", hover_name="manufacturer",animation_frame="year", 
            animation_group="manufacturer", range_y=[500, 37000], range_x=[0,280000],
            labels=dict(odometer="Odometer", price="Price", year="Release Year", manufacturer="Manufacturer", drive="Drive"), 
            category_orders={"year" : years}, template="plotly_white")
        st.subheader("Prices vs Odometer over Vehicle Release Years")
        st.write(fig)

#https://craigslist-used-cars-app.herokuapp.com/