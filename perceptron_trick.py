# Perceptron trick - page

import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from typing import Tuple


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

    








def _take_inputs_according_to_equation(equation_type: str):
    """
    This function takes input form user according to the equation (selected by user)
    """

    fetch_latex = {
        "General equation": "Ax + By + C = 0",
        "Straight line": "y = mx + c",
        "Horizontal line": "y = b",
        "Vertical line": "x = a"
    }

    line_input_col1, line_input_col2, line_input_col3 = st.columns([1,1,1])

    # display appropriate latex equation
    st.latex(fetch_latex[equation_type])

    # taking input A (coefficient of x)
    if equation_type in ["Vertical line", "General equation"]:
        with line_input_col1:
            st.number_input("A - coefficient of X", min_value=-10, value=1, max_value=10, step=1, key="general_equation_input_a_key")

    # taking input B (coefficieent of y)
    if equation_type in ["Horizontal line", "General equation", "Straight line"]:
        with line_input_col2:
            st.number_input("B - coefficient of Y", min_value=-10, value=1, max_value=10, step=1, key="general_equation_input_b_key")

    # taking input C (constant)
    if equation_type == "General equation":
        with line_input_col3:
            st.number_input("C - Constant", min_value=-10, value=1, max_value=10, step=1, key="general_equation_input_c_key")

    # taking input m (slope)
    if equation_type == "Straight line":
        with line_input_col2:
            st.number_input("m - slope", min_value=-10.0, value=1.0, max_value=10.0, step=0.1, key="general_equation_input_m_key")
        

    
        




def _get_values_for_appropriate_equation(equation: str) -> Tuple:
    
    # declaring variables
    a, b, c, m = None, None, None, None 

    # getting value of A (coefficient of x)
    if equation in ["Vertical line", "General equation"]:
        a = st.session_state.general_equation_input_a_key

    # getting value of B (coefficieent of y)
    if equation in ["Horizontal line", "General equation", "Straight line"]:
        b = st.session_state.general_equation_input_b_key

    # getting value of C (constant)
    if equation == "General equation":
        c = st.session_state.general_equation_input_c_key

    # getting value of m (slope)
    if equation == "Straight line":
        m = st.session_state.general_equation_input_m_key

    return (a, b, c, m)

        











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
    st.sidebar.toggle("Clean Graph", value=False, key="hide_all_key")
    st.sidebar.toggle("Show Grid", value=True, key="grid_key")
    st.sidebar.toggle("Show legend", value = True, key="legend_choice")
    st.sidebar.toggle("Show axis ticks", value=True, key="axis_ticks_key")





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


    # allocating spaces for graph
    space_for_graph = st.empty()


    
    # for line 1
    st.divider()
    line_input_col1, line_input_col2, line_input_col3 = st.columns([1,1,1])
    with line_input_col1:
        st.text_input("", placeholder="label for line 1", key="line_1_label_key")
    
    with line_input_col2:
        st.selectbox("Choose color", ["Red","Orange","Yellow","Brown","Cyan","Blue","Black","White"], key="colors_for_line_key")

    with line_input_col3:
        st.selectbox("", options=["General equation", "Straight line","Horizontal line","Vertical line"], key="equations_variety_key")

    # getting user selected equation
    selected_equation = st.session_state.equations_variety_key

    # taking user input according to equation selection
    _take_inputs_according_to_equation(selected_equation)

    # getting values from input widgets according to equation selection
    a, b, c, m = _get_values_for_appropriate_equation(selected_equation)

    x = np.array([-100+i for i in range(1,201)])

    # calculating y according to the equation
    match(selected_equation):
        case "General equation":
            y = -(a/b)*x - c 

        case "Straight line":
            y = m*x + c 

        case "Horizontal line":
            y = b 
        
        case "Vertical line":
            x = a







    # ----------------- Visualizing graph ------------------ #y
    if selected_equation == "Vertical line":
        plt.plot(x)
    else:
        plt.plot(x,y, color=st.session_state.colors_for_line_key.lower(), label=st.session_state.line_1_label_key)

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
        # clearing ticks
        plt.xticks([])
        plt.yticks([])

    
    
    # adding vertical line in data co-ordinates 
    plt.axvline(0, c='black', ls='--') 
    # adding horizontal line in data co-ordinates 
    plt.axhline(0, c='black', ls='--') 
     
    
    # clean graph
    if st.session_state.hide_all_key:
        plt.axis('off')

  
  
    # visualizing the mapping from values to colors 
    # plt.colorbar() 
    space_for_graph.pyplot(plt)


    
    