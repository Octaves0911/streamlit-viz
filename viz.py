from turtle import onclick
from xarray import align
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time
from streamlit_plotly_events import plotly_events
from nltk.corpus import stopwords
from collections import Counter
from nltk import ngrams
import time

st.set_page_config(layout='wide')

filter_year = None
all_stopwords = stopwords.words('english')
col1, col2 = st.columns([4, 1])
col1.subheader("Disaster Paper Visualization")
col2.subheader("Unigrams and Bigrams")

@st.cache
def init_data() -> pd.DataFrame:
    df = pd.read_csv('datav5.csv')

    return df
df = init_data()
years = df.publish_year.unique()

def init_slider(fig_slider):

    for i in range(1990,2023):
        df_filter = df.loc[df['publish_year'] < i]
        fig_slider.add_trace(go.Scatter(visible = False, x = df_filter['ebm1'], y = df_filter['ebm2'],  mode = 'markers', marker_color = df_filter['color_code'], opacity = 1, text = df_filter['title'], customdata=df_filter['authors'], hovertemplate = 'Title: %{text} <br>' + 'Author: %{customdata}'))
    fig_slider.add_trace(go.Scatter(visible = False, x = df_filter['ebm1'], y = df_filter['ebm2'],  mode = 'markers', marker_color = df_filter['color_code'], opacity = 0.2, text = df_filter['title'], customdata=df_filter['authors'], hovertemplate = 'Title: %{text} <br>' + 'Author: %{customdata}'))


    fig_slider.data[10].visible = True
    fig_slider.data[-1].visible = True

    return fig_slider

def fig_trace_update(fig):
        fig.update_traces(marker_size=3 )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        fig.update_layout(
            showlegend=False,
            template = "plotly_white",
            autosize=True,
            # width=1000,
            height=650,
            margin=dict(l=0,r=0,b=0,t=0,pad=0
            ),
         )
        return fig

def main_viz():
    
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x = df['ebm1'], y = df['ebm2'],  mode = 'markers', marker_color = df['color_code'], opacity = 1, text = df['title'], customdata=df['authors'], hovertemplate = 'Title: %{text} <br>' + 'Author: %{customdata}'))
    fig_main = fig_trace_update(fig_main)

    with col1:
        selected_data = plotly_events(
            fig_main,
            select_event= True
        )
    display_df, filter_data = get_ngrams(selected_data,df)
            

    # if len(display_df):
    #     with col2:
    #         st.dataframe(display_df)
    #         st.dataframe(filter_data)
    # else:
    #     col2.write('Select a range of papers by drawing a cluster rectangle (hold and drag mouse) on top of the projection landscape.')
    st.session_state.display_df = display_df

def search(key):
    if len(key.split()) > 1:
        paper_idx = []
        for idx,i in enumerate(df['title_auth']):
            sing_split = i.split()
            pair_text = [f'{sing_split[i]} {sing_split[i+1]}' for i in range(len(sing_split) -1 )]

            for j in pair_text:
                if key in j:
                    paper_idx.append(idx)
        paper_idx = set(paper_idx)

        if len(paper_idx) == 0:
            st.error("No match Found")
            return

    else:
        paper_idx = []
        for idx,i in enumerate(df['title_auth']):
            for j in i.split():
                if key in j:
                    paper_idx.append(idx)
        paper_idx = set(paper_idx)
        if len(paper_idx) == 0:
            st.error("No match Found")
            
            return

    filter_data_search = df.filter(items = paper_idx, axis = 0)
    fig_search = go.Figure()
    fig_search.add_trace(go.Scatter( x = filter_data_search['ebm1'], y = filter_data_search['ebm2'],  mode = 'markers', marker_color = filter_data_search['color_code'], opacity = 1, text = filter_data_search['title'], customdata=filter_data_search['authors'], hovertemplate = 'Title: %{text} <br>' + 'Author: %{customdata}'))
    
    fig_search = fig_trace_update(fig_search)

    with col1:  
        selected_data_search = plotly_events(
            fig_search,
            select_event= True
        )
    display_df, filter_data = get_ngrams(selected_data_search,df)
    # if len(display_df):
    #     with col2:
    #         pass
    #         # st.dataframe(display_df)
    #         # st.dataframe(filter_data)
    # else:
    #     col2.write('Select a range of papers by drawing a cluster rectangle (hold and drag mouse) on top of the projection landscape.')
    st.session_state.display_df = display_df

    
    
    


def get_ngrams(selected_data, filter_df):
    selected_paper = [el['pointIndex'] for el in selected_data]
    filter_data = filter_df.filter(items = selected_paper, axis = 0)

    filter_title = " ".join([x for x in filter_data['title_auth']])
    tokens_without_sw = [word.lower() for word in filter_title.split() if not word.lower() in all_stopwords]
    bigram_count = Counter(ngrams(tokens_without_sw, 2))
    unigram_count = Counter(ngrams(tokens_without_sw, 1))
    filter_data.reset_index(drop = True, inplace=True)
    display_df = pd.DataFrame()
    display_df['unigrams'] = [i[0][0] for i in unigram_count.most_common(10)]
    display_df['bigrams'] = [ f'{i[0][0]} {i[0][1]}' for i in bigram_count.most_common(10)]

    return display_df,filter_data[['title','authors']]

def year_filter_graph():
    fig_slider = go.Figure()
    fig_slider = init_slider(fig_slider)

    steps = []
    for i in range(len(fig_slider.data) - 1):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_slider.data)},
                {"title": "Slider switched to year: " + str( 1990 + i)}], 
            label=str(1990 + i)# layout attribute
        )
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][-1] = True# Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Until Year: "},
        pad={"t": 50},
        steps=steps,
        visible = True
    )]

    fig_slider.update_layout(
        sliders=sliders
    )

    fig_slider = fig_trace_update(fig_slider)

    # st.plotly_chart(fig_slider)
    with col1:
        selected_data = plotly_events(
            fig_slider,
            select_event= True
        )



    display_df, filter_data = get_ngrams(selected_data,df)

    # if len(display_df):
    #     with col2:
    #         st.dataframe(display_df)
    #         st.dataframe(filter_data)
    # else:
    #     col2.write('Select a range of papers by drawing a cluster rectangle (hold and drag mouse) on top of the projection landscape.')
    st.session_state.display_df = display_df


# if len(st.session_state.display_df.unigrams):
#     st.session_state.bt_plot = True

with col2:
        st.session_state.key = st.text_input("Enter the keyword", placeholder = "Search")
        st.session_state.bt = st.button("search")

        if st.session_state.bt or st.session_state.key:
            search(st.session_state.key)
    
with col1:
    #print(f"out  {st.session_state.bt} {st.session_state.key} ")
    
    if not st.session_state.bt and not st.session_state.key: #and not st.session_state.bt_plot:
  
        filter_year = st.checkbox("Filter by Year")
        if filter_year is False:

            main_viz()
        else:

            year_filter_graph()

if len(st.session_state.display_df.unigrams):
    st.subheader("Unigrams")
    def search_onclick(key):
        st.session_state.bt_plot = True
        search(key)

    btcol1, btcol2, btcol3, btcol4, btcol5,btcol6, btcol7,btcol8, btcol9,btcol10 = st.columns([1,1,1,1,1,1,1,1,1,1])
    btcol = [btcol1, btcol2, btcol3, btcol4, btcol5,btcol6, btcol7,btcol8, btcol9,btcol10]
    for i, uni in zip(btcol,st.session_state.display_df.unigrams):
        with i:
            st.button(uni, on_click= search_onclick, args=(uni,))
            
    st.subheader("Bigrams")
    btcol1, btcol2, btcol3, btcol4, btcol5,btcol6, btcol7,btcol8, btcol9,btcol10 = st.columns([1,1,1,1,1,1,1,1,1,1])
    btcol = [btcol1, btcol2, btcol3, btcol4, btcol5,btcol6, btcol7,btcol8, btcol9,btcol10]
    for i, uni in zip(btcol,st.session_state.display_df.bigrams):
        with i:
            st.button(uni, on_click= search_onclick, args=(uni,))
            #st.button(uni)