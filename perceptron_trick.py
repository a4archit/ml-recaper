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
                random_state=1)
    
    X_sample, y_sample = sample_data # extracting input and output


    # creating dataframes of input and output sample datasets
    X_sample_df = pd.DataFrame(X_sample, columns=['x1', 'x2'])
    y_sample_df = pd.DataFrame(y_sample, columns=['y'])


    # showing sample dataframes
    sample_data_col1, sample_data_col2 = st.columns([1,3])

    with sample_data_col1:
        st.write(X_sample_df.head(5))

    with sample_data_col2:
        st.write(y_sample_df.head(5).to_numpy())



    st.markdown("We have only two data inputs in this sample data")


    x = np.linspace(-100, 100)
    m = 0.5
    c = 1
    
    y = (m*x) + c

    # playground
    sns.scatterplot(
        X_sample_df.head(user_sample_limit), 
        legend = st.session_state.legend_choice
    )
    plt.plot(x, y, "r", scalex=False, scaley=False)
    
    st.pyplot(plt)

    



    
    