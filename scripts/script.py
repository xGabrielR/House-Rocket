import geopandas

import pandas    as pd
import numpy     as np
import streamlit as st
import folium    as fl
import plotly.express as px

from datetime import datetime
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

def setFeature(df_raw):
	df_raw['price_m2'] = df_raw['price'] / df_raw['sqft_lot']

	cols = ['sqft_living15', 'sqft_lot15']
	df_raw = df_raw.drop( cols, axis=1 )
	
	return df_raw

def overviewData(df_raw):
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
	
	return None
	

def densityMap(df_raw, gf):
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

	return None


def commercialCategory(df_raw):
	st.sidebar.title('Commercial Options')
	st.title('Commercial Attributes')

	df_raw['date'] = pd.to_datetime( df_raw['date'] ).dt.strftime('%Y-%m-%d')

	# Average Price per Year
	st.sidebar.subheader('Max Year Filter')

	st.header('Average Price per Year Built')

	# Filters
	min_yr_built = int( df_raw['yr_built'].min() )
	max_yr_built = int( df_raw['yr_built'].max() )

	fill_year = st.sidebar.slider( 'Year Build', min_yr_built, max_yr_built, min_yr_built )

	dfx = df_raw.loc[df_raw['yr_built'] < fill_year]
	yrPrice = dfx[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

	fig = px.line( yrPrice, x='yr_built', y='price' )
	st.plotly_chart( fig, use_container_width=True )

	# Average Price per Day

	st.header('Average Price per Day')
	st.sidebar.subheader('Select Date')

	# Filters
	min_date = datetime.strptime( df_raw['date'].min(), '%Y-%m-%d' )
	max_date = datetime.strptime( df_raw['date'].max(), '%Y-%m-%d' )

	fill_date = st.sidebar.slider( 'Date / Day', min_date, max_date, min_date )

	df_raw['date'] = pd.to_datetime( df_raw['date'] )
	dfz = df_raw.loc[df_raw['date'] < fill_date] 
	datePrice = dfz[['date', 'price']].groupby('date').mean().reset_index()

	fig = px.line( datePrice, x='date', y='price' )
	st.plotly_chart( fig, use_container_width=True )

	# Histogram's
	# -----------------------
	st.header('Price Distribuition')
	st.sidebar.subheader('Select Max Price')

	# Filters
	min_price = int(df_raw['price'].min() )
	max_price = int(df_raw['price'].max() )
	avg_price = int(df_raw['price'].mean() )

	fill_price = st.sidebar.slider( 'Price', min_price, max_price, avg_price )
	dfp = df_raw.loc[df_raw['price'] < fill_price ]

	fig = px.histogram( dfp, x='price', nbins=50 )
	st.plotly_chart( fig, use_width_container=True )

	return None
	

def otherCategory(df_raw):
	st.sidebar.title('Attributes Options')

	fill_bed = st.sidebar.selectbox( 'Max Number of Bedrooms', sorted( set( df_raw['bedrooms'].unique() ) ) )

	fill_bath = st.sidebar.selectbox( 'Max Number of Bathrooms', sorted ( set( df_raw['bathrooms'].unique() ) ) )

	st.beta_columns(2)

	# Bedrooms
	c1.header('Houses per Bedrooms')
	dfb = df_raw[df_raw['bedrooms'] < fill_bed]
	fig = px.histogram( dfb, x='bedrooms', nbins=20 )
	c1.plotly_chart( fig, use_width_container=True )

	# Bathrooms
	c2.header('Houses per Bathrooms')
	dfn = df_raw[df_raw['bathrooms'] < fill_bath]
	fig = px.histogram( dfn, x='bathrooms', nbins=20 )
	c2.plotly_chart( fig, use_width_container=True )

	# filters
	fill_floors = st.sidebar.selectbox( 'Max Number of Floors', sorted ( set( df_raw['floors'].unique() ) ) )

	fill_wather = st.sidebar.checkbox( 'Only with Waterview' )

	c1, c2 = st.beta_columns(2)

	# Floors
	c1.header('Houses per Floors')
	dff = df_raw[df_raw['floors'] < fill_floors]
	fig = px.histogram( dff, x='floors', nbins=10 )
	c1.plotly_chart( fig, use_width_container=True )

	# Water View
	c2.header('Houses with Waterfront')
	if fill_wather:
		dfw = df_raw[df_raw['waterfront'] == 1]
	else:
		dfw = df_raw.copy()

	fig = px.histogram( dfw, x='waterfront', nbins=5 )
	c2.plotly_chart( fig, use_width_container=True )


return None


def __name__ == '__main__':
	# Data Extraction
	path = "../data/kc_house_data.csv"
	url = "https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson"
	
	df_raw = getData(path)
	gf = getGeoData(url)
	
	
	# Data Transformation
	df_raw = setFeature(df_raw)
	
	overviewData(df_raw)
	
	densityMaps(df_raw, gf)
	
	commercialCategory(df_raw)
	
	otherCategory(df_raw)
	
	# Data Loading //

