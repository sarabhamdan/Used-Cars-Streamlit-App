import pandas as pd
import numpy as np
import scipy as sp
#import opendatasets as od
from datetime import datetime
import time

import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import iplot
import plotly.graph_objects as go
from PIL import Image

import streamlit as st

st.title("Used Cars on Craigslist")

mapbox_access_token= "pk.eyJ1Ijoic2FyYS1iaGFtZGFuIiwiYSI6ImNrdGJ4eGVzNTIwdTUybnBkeXVibzI4ODQifQ.l34h5B7ThJQc-nRlHB-eHg"

@st.cache
def load_data():
    df = pd.read_csv('vehicles.csv')
    #drop unneeded columns
    df.drop(['url','region_url', 'image_url', 'county','description','model','VIN', 'id', 'posting_date'],axis=1, inplace=True)
    df.dropna(inplace=True)

    # calculate interquartile range for price
    q25_prc, q75_prc = np.nanpercentile(df.price, 25), np.nanpercentile(df.price, 75)
    iqr_prc = q75_prc - q25_prc
    # calculate the outlier cutoff for price
    cut_off_prc = iqr_prc * 1.5
    lower_prc, upper_prc = q25_prc - cut_off_prc, q75_prc + cut_off_prc
    # calculate interquartile range for odometer
    q25_odo, q75_odo = np.nanpercentile(df.odometer, 25), np.nanpercentile(df.odometer, 75)
    iqr_odo = q75_odo - q25_odo
    # calculate the outlier cutoff for odometer
    cut_off_odo = iqr_odo * 1.5
    lower_odo, upper_odo = q25_odo - cut_off_odo, q75_odo + cut_off_odo
    # calculate interquartile range for year
    q25_yr, q75_yr = np.nanpercentile(df.year, 25), np.nanpercentile(df.year, 75)
    iqr_yr = q75_yr - q25_yr
    # calculate the outlier cutoff for year
    cut_off_yr = iqr_yr * 1.5
    lower_yr, upper_yr = q25_yr - cut_off_yr, q75_yr + cut_off_yr
    # #remove outliers
    df=df[(df.price > lower_prc) & (df.price < upper_prc)]
    df=df[(df.odometer > lower_odo) & (df.odometer < upper_odo)]
    df=df[(df.year > lower_yr) & (df.year < upper_yr)]
    df.reset_index(drop=True,inplace=True)

    df['state']=df.state.str.upper()
    
    df.condition= df.condition.str.replace("like new", "new")
    df.condition= df.condition.str.replace("good", "fair")
    df['condition']=df.condition.str.capitalize()

    # remove cars with price < $500 
    df.drop(df[(df.price < 500)].index, inplace = True)
    df.reset_index(drop=True, inplace=True)

    # Dropping cars with price less than 1000 with miles less than 60,000 and model year greater than 2010
    df.drop(df[(df.price < 1000 ) & (df.odometer < 60000 ) & (df.year > 2010)].index, inplace = True)

    #convert year to int
    df['year']=df.year.astype(int)

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
    st.markdown("""
        Craigslist is the world's largest collection of used vehicles for sale. This data is scraped every few months, 
        it contains most all relevant information that Craigslist provides on car sales including columns like price, condition, 
        manufacturer, latitude/longitude, and 18 other categories. 
        """)
    st.markdown("The original dataset can be found on Kaggle: https://www.kaggle.com/austinreese/craigslist-carstrucks-data/version/10")

    img=Image.open('cars.jfif')
    st.image(img,width=600)

    # Visualize Car conditions
    st.subheader("Condition of Used Cars for Sale on Craigslist")
    condition_pie=px.pie(df, "condition", labels=dict(condition="Condition"))
    st.write(condition_pie)

# Visualize Raw Data
if st.sidebar.checkbox("Dataset Quick Look"):
    st.subheader("Dataset Quick Look")
    st.write(df.head(10))

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
if st.sidebar.checkbox("Used Car Locations"):
    st.subheader("Location of Used Cars on Craigslist")
    option=st.multiselect("State", df['state'].unique(), default=df['state'].unique())
    
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
if st.sidebar.checkbox("Used Car Prices by Manufacturer"):
    st.subheader("Used Car Prices per Top 10 Manufacturers over Release Years")
    
    #select top 10 manufacturers
    freq_manufacturers= list(df.manufacturer.value_counts().to_frame().head(10).index)
    df3=df[df['manufacturer'].isin(freq_manufacturers)]
    
    years=list(df3.year.unique())
    years.sort()

    release_year= st.slider('Car Release Year',1992,2022, 2020)

    fig = px.bar(df3[df3.year == release_year], x="manufacturer", y="price", color='manufacturer',
             labels=dict(manufacturer= "Manufacturer", price= 'Price', year="Release Year", opacity= 50,template='plotly_light')) 
             #animation_frame= "year", animation_group="manufacturer", range_y= [500, 38000], category_orders={"year" : years}, )
    st.write(fig)
    if st.button("Check Box Plot of Price for Top 10 Manufacturers"):
        fig= px.box(df3, x="manufacturer", y="price", color='manufacturer', labels=dict(manufacturer= "Manufacturer", price= 'Price'))
        st.write(fig)

# Visualize relationship of price with odometer and year
if st.sidebar.checkbox("Correlation Analysis of Used Car Prices"):
    st.subheader("Relationship of Price with Release Year & Odometer")
    fig= px.scatter_matrix(df, dimensions=["price", "year", "odometer"], color="size", opacity=0.16, 
        labels=dict(price= "Price", odometer="Odometer", year="Release Year", size="Car Size"))
    st.write(fig)
    st.markdown('As expected, a positive relationship is noted between release year and used car price versus a negative relationship between odometer and used car price.')
    st.markdown("For more on price vs odometer and release year, select the last 2 checkboxes in the side panel.")
    if st.sidebar.checkbox("Price vs Odometer by Used Car Type"):
        freq_types=list(df['type'].value_counts().to_frame().head(3).index)
        df2= df[df['type'].isin(freq_types)]
        fig= px.scatter(df2[:3000], x= "odometer", y="price", color= "drive", size= "year", size_max=3.5, hover_name="manufacturer", facet_col= "type", 
            labels=dict(price= "Price", odometer="Odometer", drive="Drive", type="Type"))
        st.subheader("Price vs Odometer by Used Car Type")
        st.write(fig)
            
    if st.sidebar.checkbox("Prices vs Odometer over Car Release Years"):
        years=list(df.year.unique())
        years.sort()
        fig= px.scatter(df, x="odometer", y="price", color= "drive", hover_name="manufacturer",animation_frame="year", 
            animation_group="manufacturer", range_y=[500, 37000], range_x=[0,280000],
            labels=dict(odometer="Odometer", price="Price", year="Release Year", manufacturer="Manufacturer", drive="Drive"), 
            category_orders={"year" : years}, template="plotly_white")
        st.subheader("Prices vs Odometer over Car Release Years")
        st.write(fig)

