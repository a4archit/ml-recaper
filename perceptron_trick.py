# Perceptron trick - page

import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs


def perceptron_trick_main():

    # -------------------------- Sidebar ------------------- #
    # st.sidebar.header("Perceptron Trick ")
    st.sidebar.divider()
    st.sidebar.subheader("Graph settings")
    # visualization customization
    user_sample_limit = st.sidebar.slider(
        label = "Number of data points",
        min_value=10, 
        max_value=100,
        step=10,
        value = 40
    )
    st.sidebar.toggle("Show labels", value = True, key="legend_choice")


    about_perceptron_trick = f"""
        ### Perceptron Trick
        Perceptron is a mimic of human neuron that would be able to understand as like humans
        and able to find out the patterns from data. It is a mathematical model. 
        A single perceptron takes some numbers as input and performing some operations on them 
        afterthat return some number(s) as output.

    """
    st.markdown(about_perceptron_trick)
    # st.latex("\hat y = \sum_{i=0}^n(w_ix_i) + b  ")
    formula_y_pred = """
        \hat y = \sum_{i=0}^n(w_ix_i) + b
        \\\[.2in]
        
        \\\ b       \\rightarrow bias
        \\\ w_i     \\rightarrow weights
        \\\ \hat y  \\rightarrow prediction 
        \\\ x_i     \\rightarrow current\ input
        \\\ n       \\rightarrow number\ of\ inputs
        \\\ i       \\rightarrow current\ input\ iteration 
        
    """
    st.latex(formula_y_pred)

    st.image(
        "https://miro.medium.com/v2/resize:fit:1400/1*dsVvCeoxlU4GZ1y701Mo8g.png",
        use_container_width=True,
        caption = "Perceptron")
    
    st.markdown("""
                #### Taking sample data 
                In this sample data we have two classes (0 and 1)
    """, unsafe_allow_html=True)


    # Generate a linearly separable dataset with two classes
    sample_data = make_blobs(
                n_features=2,
                centers=1, 
                cluster_std = 0.5,
                random_state=0)
    
    X_sample, y_sample = sample_data # extracting input and output


    # creating dataframes of input and output sample datasets
    X_sample_df = pd.DataFrame(X_sample, columns=['x1', 'x2'])
    y_sample_df = pd.DataFrame(y_sample, columns=['y'])

    # making data for 4 quadrants
    X_sample_df.x1 = X_sample_df.x1.apply(lambda x: x-1)
    X_sample_df.x2 = X_sample_df.x2.apply(lambda x: x-1)


    # showing sample dataframes
    sample_data_col1, sample_data_col2 = st.columns([1,3])

    with sample_data_col1:
        st.write(X_sample_df.head(5))

    with sample_data_col2:
        st.write(y_sample_df.head(5).to_numpy())



    st.markdown("We have only two data inputs in this sample data")


    x = np.linspace(-100, 100)
    m = 0.5
    c = 2
    
    y = (m*x) + c

    # ----------------- playground ------------------ #
    sns.scatterplot(
        X_sample_df.sample(user_sample_limit), 
        legend = st.session_state.legend_choice
    )
    # plt.plot(x, y, "r", scalex=False, scaley=False)
    plt.xlim(-10, 40)
    plt.ylim(-4, 8)
    # adding style theme in scatter plot 
    
    # adding vertical line in data co-ordinates 
    plt.axvline(0, c='black', ls='--') 
    
    # adding horizontal line in data co-ordinates 
    plt.axhline(0, c='black', ls='--') 
    
    # giving x label to the plot 
    plt.xlabel("X axis") 
    
    # giving y label to the plot 
    plt.ylabel("Y axis") 
    
    # giving title to the plot 
    plt.title("Sample data visualization") 
  
  
    # visualizing the mapping from values to colors 
    # plt.colorbar() 
    st.pyplot(plt)

    













def perceptron_trick_playground():
    # ------------------- Graph Title ------------------ #
    # st.sidebar.header("Perceptron Trick ")
    st.sidebar.title("Graph settings")
    st.sidebar.divider()

    # title settings
    st.sidebar.text_input("Title", placeholder="Graph title", key="graph_title_input_key")
    st.sidebar.toggle("Show", value=True, key="graph_title_key")
    
    # X axis label
    st.sidebar.divider()
    st.sidebar.markdown("### Axis labels")
    st.sidebar.text_input("X - axis label", placeholder="X axis label", key="x_axis_label_key")
    st.sidebar.text_input("Y - axis label", placeholder="Y axis label", key="y_axis_label_key")
    st.sidebar.toggle("Show", value=True, key="axis_labels_key")

    # axis limits
    st.sidebar.divider()
    st.sidebar.markdown("### Axis limits")
    st.sidebar.number_input("X - axis min", key="x_limit_min_key", min_value=-1000, max_value=1000, step=10, value=0)
    st.sidebar.number_input("X - axis max", key="x_limit_max_key", min_value=-1000, max_value=1000, step=10, value=40)
    st.sidebar.number_input("Y - axis min", key="y_limit_min_key", min_value=-1000, max_value=1000, step=2, value=-4)
    st.sidebar.number_input("Y - axis max", key="y_limit_max_key", min_value=-1000, max_value=1000, step=2, value=8)
    st.sidebar.toggle("Use by default limits", value=False, key="axis_limits")


    # datapoints
    st.sidebar.divider()
    user_sample_limit = st.sidebar.slider(label = "Number of data points",min_value=10, max_value=100,step=10,value = 40)
    st.sidebar.toggle("Show datapoints", value=True, key="datapoints_key")


    # other settings
    st.sidebar.divider()
    st.sidebar.toggle("Grid", value=True, key="grid_key")
    st.sidebar.toggle("Show legend", value = True, key="legend_choice")
    st.sidebar.toggle("Show ticks", value=True, key="axis_ticks_key")





    # ----------------- Creating data --------------- #

    # Generate a linearly separable dataset with two classes
    sample_data = make_blobs(
                n_features=2,
                centers=1, 
                random_state=1)

    X_sample, y_sample = sample_data # extracting input and output


    # creating dataframes of input and output sample datasets
    X_sample_df = pd.DataFrame(X_sample, columns=['x1', 'x2'])
    y_sample_df = pd.DataFrame(y_sample, columns=['y'])

    # making data for 4 quadrants
    X_sample_df.x1 = X_sample_df.x1.apply(lambda x: x-0.3)
    X_sample_df.x2 = X_sample_df.x2.apply(lambda x: x+1)


    x = np.linspace(-100, 100)
    m = 0.5
    c = 2
    
    y = (m*x) + c



    # ----------------- playground ------------------ #

    # giving title to the plot 
    if st.session_state.graph_title_key:
        title = st.session_state.graph_title_input_key
        plt.title("Custom title" if title == "" else title) 


    # plotting graph 
    if st.session_state.datapoints_key:
        sns.scatterplot(
            X_sample_df.head(user_sample_limit), 
            legend = st.session_state.legend_choice
        )

    # axis labels
    if st.session_state.axis_labels_key:
        x_label = st.session_state.x_axis_label_key
        y_label = st.session_state.y_axis_label_key
        plt.xlabel("X axis" if x_label == "" else x_label) 
        plt.ylabel("y axis" if y_label == "" else y_label)


    # axis limits
    if not st.session_state.axis_limits:
        plt.xlim(st.session_state.x_limit_min_key, st.session_state.x_limit_max_key)
        plt.ylim(st.session_state.y_limit_min_key, st.session_state.y_limit_max_key)


    if st.session_state.grid_key:
        plt.grid()  

    
    # axis ticks
    if not (st.session_state.axis_ticks_key):
        plt.xticks([])
        plt.yticks([])

    
    
    # adding vertical line in data co-ordinates 
    plt.axvline(0, c='black', ls='--') 
    # adding horizontal line in data co-ordinates 
    plt.axhline(0, c='black', ls='--') 
     
    

    plt.axis('on')

  
  
    # visualizing the mapping from values to colors 
    # plt.colorbar() 
    st.pyplot(plt)

    st.sidebar.header("Configurations")

    
    