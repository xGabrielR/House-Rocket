import geopandas
import pandas    as pd
import numpy     as np
import streamlit as st
import folium    as fl 
from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster


# Set Wide Page (Show at streamlit app)
st.set_page_config( layout='wide' )

@st.cache( allow_output_mutation=True )
def getData(path):
    data = pd.read_csv(path)
    return data

@st.cache( allow_output_mutation=True )
def getGeoData(url):
    gf = geopandas.read_file(url)
    return gf

# Get Data
path = "../data/kc_house_data.csv"
df_raw = getData(path)

# Get geofile
url = "https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson"
gf = getGeoData(url)

# Add new Features & Data Clearing
df_raw['price_m2'] = df_raw['price'] / df_raw['sqft_lot']

cols = ['sqft_living15', 'sqft_lot15']
df_raw = df_raw.drop( cols, axis=1 )

#df_raw = pd.to_datetime(df_raw['date'])

# --------------------------
# Data Overview
# --------------------------
fill_attributes = st.sidebar.multiselect( 'Select Filters', df_raw.columns )
fill_zipcode = st.sidebar.multiselect('Enter Zipcode', df_raw['zipcode'].unique() )


st.title('Data Overview')

# Attributes + Zipcode = Select Lines and Columns
# Attributes = Select Columns
# Zipcode = Select Lines
# Null + Null = Return Original Dataset


if ( fill_zipcode != [] ) & ( fill_attributes != [] ):
    df_raw = df_raw.loc[df_raw['zipcode'].isin( fill_zipcode ), fill_attributes]

elif ( fill_zipcode != [] ) & ( fill_attributes == [] ):
    df_raw = df_raw.loc[df_raw['zipcode'].isin( fill_zipcode ), : ]

elif ( fill_zipcode == [] )& ( fill_attributes != []):
    df_raw = df_raw.loc[:, fill_attributes ]

else:
    df_raw = df_raw.copy()


st.write( fill_attributes )
st.write( fill_zipcode )


st.write(df_raw)

# Avg Metrics
df1 = df_raw[['id', 'zipcode']].groupby('zipcode').count().reset_index()
df2 = df_raw[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df3 = df_raw[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
df4 = df_raw[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

c1, c2 = st.beta_columns((1, 1))

# Merge Metrics
m1 = pd.merge( df1, df2, on='zipcode', how='inner' )
m2 = pd.merge( m1, df3, on='zipcode', how='inner' )
df = pd.merge( m2, df4, on='zipcode', how='inner' )

c1.header('Average Values')
c1.write( df, heigh=600 )

# Statistic Descriptive
num_att = df_raw.select_dtypes(include=['int64', 'float64'])
mean = pd.DataFrame( num_att.apply( np.mean ) )
median = pd.DataFrame(num_att.apply( np.median ) )
std = pd.DataFrame( num_att.apply( np.std ))
max_ = pd.DataFrame( num_att.apply( np.max ) )
min_ = pd.DataFrame( num_att.apply( np.min ) )

df1 = pd.concat([ max_, min_, mean, median, std ], axis=1 ).reset_index()
df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

c2.header('Statistic Descriptive')
c2.dataframe( df1, height=300 )


# ----------------------
# Density Maps
# ----------------------

st.title('Region Overview')

c1, c2 = st.beta_columns((1, 1))
c1.header('Density Map')

df = df_raw.sample(10)

# Base Map - Folium

density_map = fl.Map( location=[df_raw['lat'].mean(), 
                      df_raw['long'].mean()], 
                      default_zoom_start=15 )

marker_cluster = MarkerCluster().add_to(density_map)
for name, row in df.iterrows():
    fl.Marker( [row['lat'], row['long']],
            popup='Sold R${0} on: {1}, Sqft: {2}, Bedrooms: {3}'.format( row['price'], row['date'], row['sqft_living'], row['bedrooms'])).add_to(marker_cluster)

with c1:
    folium_static(density_map)


# Region Price Map

c2.header('Price Density')

df = df_raw[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df.columns = ['ZIP', 'PRICE']

df = df.sample( 10 )

gf = gf[gf['ZIP'].isin(df['ZIP'].tolist())]

region_map = fl.Map( location=[df_raw['lat'].mean(), 
                     df_raw['long'].mean() ], 
                     default_zoom_start=15 )

region_map.choropleth( data = df,
                       geo_data=gf,
                       columns=['ZIP', 'PRICE'],
                       key_on='feature.properties.ZIP',
                       fill_color='YlOrRd',
                       fill_opacity=0.7,
                       line_opacity=0.2,
                       legend_name='AVG PRICE')

with c2:
    folium_static(region_map)

# -------------------
# 
# ---------------------
