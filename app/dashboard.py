import folium
import pathlib
import warnings
import geopandas
import numpy as np
import pandas as pd
import streamlit as st
import plotly_express as px
from datetime import datetime, timedelta
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static 

st.set_page_config(page_title="| HR Dashboard", page_icon="🏠", layout="wide")
warnings.filterwarnings("ignore")

@st.cache(allow_output_mutation=True)
def load_dataset(path):
    df = pd.read_csv(path)
    return df

@st.cache(allow_output_mutation=True)
def load_geo(url):
    geofile = geopandas.read_file(url)
    return geofile

class HouseRocketDashboard():
    def __init__(self):
        self.img_path = "img.png" # Do Not Need str(pathlib.Path().resolve()).replace("\\", "/") + "/img.png"
    
    def css_template(self, ):
        html = """
        <style>
            p {color: #428df5; }
            ::selection { color: #b950ff; }
            h1 {color: #7033ff; text-align: center; }
            h2 {color: #9d73ff}
            h3 {text-align: center;}
        </style>
        """
        st.markdown( html, unsafe_allow_html=True )
        return None

    def feature_engineering(self, df):
        df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d")
        df["date"] = df.date.dt.strftime("%Y-%m-%d")
        df["price_sqft"] = df.price / df.sqft_lot

        return df

    def get_metrics(self, df):
        num_att = df.select_dtypes(include=["int64", "float64"])
        d1 = pd.DataFrame(num_att.apply(np.mean)).T
        d2 = pd.DataFrame(num_att.apply(np.median)).T
        c1 = pd.DataFrame(num_att.apply(min)).T
        c2 = pd.DataFrame(num_att.apply(max)).T
        c3 = pd.DataFrame(num_att.apply(np.std)).T
        c4 = pd.DataFrame(num_att.apply(lambda x: x.max() - x.min())).T

        m = pd.concat([c1, c2, d1, d2, c3, c4]).T.reset_index()
        m.columns = ["atributos", "minimo", "maximo", "media", "mediana", "desvio-padrão", "amplitude"]

        m = m.iloc[1:, :].reset_index(drop=True)

        return m

    def get_static(self, df):
        aux1 = df[["id", "zipcode"]].groupby("zipcode").count().reset_index()
        aux2 = df[["price", "zipcode"]].groupby("zipcode").mean().reset_index()
        aux3 = df[["sqft_living", "zipcode"]].groupby("zipcode").mean().reset_index()
        aux4 = df[["price_sqft", "zipcode"]].groupby("zipcode").mean().reset_index()

        m1 = pd.merge(aux1, aux2, on="zipcode", how="inner")
        m2 = pd.merge(m1, aux3, on="zipcode", how="inner")
        df2 = pd.merge(m2, aux4, on="zipcode", how="inner")

        return df2.style.highlight_max(color="#08018f", subset=pd.IndexSlice[:, df2.columns.tolist()[2:]])

    def get_potential_houses(self, df):
        st.write("*Preço Médio Atual de todas as Casas*: $  {:.3f}".format(np.mean(df.price)))
        df = df[(df["condition"] >= 3) & (df["price"] <= np.mean(df.price)) & (df["floors"] >= 2)]

        return df


    def app_header(self, ):
        st.image(self.img_path)

        hour = int( datetime.now().hour )
        if (hour <= 12) & (hour >= 0):
            st.subheader('Bom Dia CEO! 😁')
        if (hour > 12) & (hour <= 17):
            st.subheader('Boa Tarde CEO! 😊')
        if (hour > 17):
            st.subheader('Boa Noite CEO! 😴')
        st.write("_____________")

        st.header("Selecione as Opções Abaixo")
        st.write(" ")

        return None

    def table_header(self, ):
        st.write("_________")
        st.header("Visão Geral")
        st.sidebar.title("◇ Filtros Gerais")
        st.sidebar.write("___")

        return None

    def plot_header(self, ):
        st.write("_________")

        st.sidebar.title("◇ Filtros de Características")
        st.sidebar.header("Atributos Comerciais")
        st.sidebar.write("_____")

        return None

    def get_page(self, ):
        page = st.selectbox("Selecione seu tipo de Visualização", ["Tabelas", "Gráficos", "Mapas"])
        
        return page

    def table_page(self, df):
        self.table_header()

        filter_features = st.sidebar.multiselect("Informe as Características", df.columns.tolist())
        filter_zipcode  = st.sidebar.multiselect("Informe um Zipcode", df.zipcode.unique())

        df2 = df.copy() # Static DF

        if (filter_zipcode == []) & (filter_features == []):
            df = df.iloc[:, :]

        elif (filter_zipcode != []) & (filter_features == []):
            df = df.loc[df["zipcode"].isin(filter_zipcode), :]

        elif (filter_zipcode == []) & (filter_features != []):
            df = df.loc[:, filter_features]

        else:
            df = df.loc[df["zipcode"].isin(filter_zipcode), filter_features]

        st.dataframe(df.head(50))

        st.write("_______")
        c1, c2 = st.columns((2))
        c1.header("Dados Estáticos")
        c2.header("Descrição Estatística")
        c1.dataframe(self.get_static(df2), height=500)
        c2.dataframe(self.get_metrics(df), height=500)

        st.write("_______")
        st.header("Potencial Casas")
        st.dataframe(self.get_potential_houses(df2))

        return None

    def plot_commercial(self, df, filter_date, filter_year, 
                            filter_sqft, filter_price):

        df["date"] = pd.to_datetime(df["date"])
        df_date = df.loc[df["date"] < filter_date]
        df_date = df_date[["price", "date"]].groupby("date").mean().reset_index()

        df_year = df.loc[df["yr_built"] < filter_year]
        df_year = df_year[["price", "yr_built"]].groupby("yr_built").mean().reset_index()

        df_lot = df.loc[df["sqft_lot"] < filter_sqft]

        df_price = df.loc[df["price"] < filter_price]

        fig = px.line(x=df_year.yr_built, y=df_year.price, title="Preço / Ano de Construção",
                    color_discrete_sequence=["red"], labels={"x": "Ano de Construção", "y": "Preço",})
        fig.update_layout( plot_bgcolor='#0e1117' )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(x=df_date.date, y=df_date.price, title="Preço / Data",
                    color_discrete_sequence=["red"], labels={"x": "Data", "y": "Preço",})
        fig.update_layout( plot_bgcolor='#0e1117' )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(x=df_price.price, nbins=400, title="Histograma de Preços",
                        color_discrete_sequence=["blue"], labels={"x": "Preço",})
        fig.update_layout( plot_bgcolor='#0e1117' )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(x=df_lot.sqft_lot, nbins=400, title="Histograma de Sqft Lote",
                        color_discrete_sequence=["blue"], labels={"x": "Sqft Lote",})
        fig.update_layout( plot_bgcolor='#0e1117' )
        st.plotly_chart(fig, use_container_width=True)

        return None

    def plot_features(self, df, filter_floors, filter_bathrooms, filter_bedrooms, 
                        filter_condition, filter_waterfront):

        df_floor = df.loc[df["floors"]    <= filter_floors]
        df_bath  = df.loc[df["bathrooms"] <= filter_bathrooms]
        df_bed   = df.loc[df["bedrooms"]  <= filter_bedrooms]
        df_cond  = df.loc[df["condition"] <= filter_condition]

        c1, c2 = st.columns((2))
        fig = px.histogram(df_bath.bathrooms, nbins=50, title="Casas por Banheiros",
                        color_discrete_sequence=["yellow"])
        fig.update_layout(plot_bgcolor="#0e1117")
        c1.plotly_chart(fig)

        fig = px.histogram(df_bed.bedrooms, nbins=50, title="Casas por Quartos",
                        color_discrete_sequence=["yellow"])
        fig.update_layout(plot_bgcolor="#0e1117")
        c2.plotly_chart(fig)

        c3, c4 = st.columns((2))
        fig = px.histogram(df_floor.floors, nbins=50, title="Casas por Andares",
                        color_discrete_sequence=["yellow"])
        fig.update_layout(plot_bgcolor="#0e1117")
        c1.plotly_chart(fig)

        fig = px.histogram(df_cond.condition, nbins=50, title="Casas pela Condição",
                        color_discrete_sequence=["yellow"])
        fig.update_layout(plot_bgcolor="#0e1117")
        c2.plotly_chart(fig)

        c5, c6 = st.columns((2))
        aux = df[["price", "waterfront"]].groupby("waterfront").median().reset_index()
        fig = px.pie(aux, values="price", names="waterfront", title="Mediana de Preço por Visão p/ Água",
                    color_discrete_sequence=["red", "black"])
        fig.update_traces(marker=dict(line=dict(color='#fff', width=2)))
        c6.plotly_chart(fig)

        if filter_waterfront:
            df_water = df.loc[df["waterfront"] == 1]
            fig = px.histogram(df_water.waterfront, nbins=20, title="Casas com Visão para Água")
            c5.plotly_chart(fig)

        else:
            fig = px.histogram(df.waterfront, nbins=20, title="Visão para Água")
            c5.plotly_chart(fig)


        return None

    def plot_page(self, df):

        self.plot_header()

        filter_year  = st.sidebar.slider("Selecione o Ano de Construção", int(df["yr_built"].min()+10), int(df["yr_built"].max()))
        filter_sqft  = st.sidebar.slider("Selecione o Tamanho do Lote", int(df["sqft_lot"].min()+10), int(df["sqft_lot"].max()))
        filter_price = st.sidebar.slider("Selecione a Amplitude do Preço", int(df.price.min()+5000), int(df.price.max()))
        filter_date  = st.sidebar.slider("Selecione a Data / Dia", (datetime.strptime(df["date"].min(), "%Y-%m-%d") + timedelta(days=3)), 
                                                                datetime.strptime(df["date"].max(), "%Y-%m-%d"))

        st.sidebar.header("Atributos Físicos")
        filter_floors     = st.sidebar.selectbox("Selecione o N° de Andares", sorted(set(df.floors.unique())))
        filter_bathrooms  = st.sidebar.selectbox("Selecione o N° de Banheiros", sorted(set(df.bathrooms.unique())))
        filter_bedrooms   = st.sidebar.selectbox("Selecione o N° de Quartos", sorted(set(df.bedrooms.unique())))
        filter_condition  = st.sidebar.selectbox("Selecione a Condição", sorted(set(df.condition.unique())))
        filter_waterfront = st.sidebar.checkbox("Casas com Visão para Água")

        st.header("Atributos Comerciais")
        self.plot_commercial(df, filter_date, filter_year, filter_sqft, filter_price)

        st.header("Atributos Físicos")
        self.plot_features(df, filter_floors, filter_bathrooms, filter_bedrooms, filter_condition, filter_waterfront)

        return None

    def map_page(self, df, geo):

        st.write("_________")

        df = df.head(10000) # FOR FOLIUM MAP

        st.header("Mapa de Densidade de Preço")

        marker_map = folium.Map(location=[df["lat"].mean(), df["long"].mean()], default_zoom_start=10)
        mc = MarkerCluster().add_to(marker_map)
        
        for name, row in df.iterrows():
            tooltip = f"<strong>Id: {row['id']}</strong>"
            text_popup = f"<b>Preço</b>: {row['price']}<br><b>Data</b>: {row['date']}<br>\
                        <b>SQFT Lote</b>: {row['sqft_lot']}<br><b>Quartos</b>: {row['bedrooms']}"

            folium.Marker([row["lat"], row["long"]], 
                        popup=folium.Popup(text_popup, max_width=170), tooltip=tooltip).add_to(mc)

        folium_static(marker_map, 1200, 700)

        st.header("Mapa de Densidade de Portfólio")

        df2 = df[["zipcode", "price"]].groupby("zipcode").mean().reset_index()
        df2.columns = ["ZIP", "PRICE"]
        geo = geo[geo["ZIP"].isin(df2["ZIP"].tolist())]

        region_map = folium.Map(location=[df["lat"].mean(), df["long"].mean()], default_zoom_start=10)

        region_map.choropleth(data=df2, geo_data=geo, 
                            columns=["ZIP", "PRICE"], 
                            key_on="feature.properties.ZIP",
                            fill_color='YlOrRd', fill_opacity=.7, line_opacity=.2,
                            legend_name="Preço Médio")

        folium_static(region_map, 1200, 700)

        return None


if __name__ == "__main__":

    hrd = HouseRocketDashboard()

    geopath = "../data/zipcodes_seattle.geojson" 
    path = "../data/kc_house_data.csv"

    df  = load_dataset(path)
    geo = load_geo(geopath)

    df = hrd.feature_engineering(df)

    hrd.css_template()
    hrd.app_header()

    page = hrd.get_page()

    if page == "Tabelas":
        hrd.table_page(df)

    elif page == "Gráficos":
        hrd.plot_page(df)

    elif page == "Mapas":
        hrd.map_page(df, geo)