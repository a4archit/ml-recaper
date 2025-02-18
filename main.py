"""

    ML Recaper
    ============

    ML Recaper is a tool that helps Machine learning learners to 
    revise concepts of machine learning quickly. It also provide
    a good, attractive and interactive graphical intuitions behind 
    the concepts.

    

    Why use ML Recaper?
    -------------------

    - It visualize data flow 
    - smooth concepts flow
    

"""



# imporing packages and utilities
import streamlit as st 
from perceptron_trick import perceptron_trick_main 
from ann import ann_main







# ------------------- initializing some parameters --------------------- #

DEVELOPING_TIME = True 

global_page_name = ""
logo_url = "https://cdn-icons-png.flaticon.com/512/7017/7017532.png"
st.set_page_config("ML Recaper", logo_url)
ml_recaper_intro = """
    
    # <center>ML Recaper</center>  
      
    *************

    ML Recaper is a tool that helps Machine learning learners to 
    revise concepts of machine learning quickly. It also provide
    a good, attractive and interactive graphical intuitions behind 
    the concepts.
"""

ml_recaper_other_intro = """
    ##### Why use ML Recaper?

    - It visualize data flow 
    - smooth concepts flow
"""












#------------------------ python functions ------------------- #

# this code same in all pages
def global_code():
    logo_col1, logo_col2 = st.columns([1,4])

    with logo_col1:
        st.image(logo_url, width=90)

    with logo_col2:
        st.title("ML Recaper")

    st.divider()





def intro():
    
    intro_description = f"""
        <p style="font-size: 1.3rem; padding: 0 5%;">
            You can boost your machine learning concepts by the use of this tool in easy and interesting way. 
            So select a topic from <b>sidebar</b> what you want to revise or study.
            <br>
            <br>
            <br>
            <b>Why use ML Recaper?</b>
            <br>
            1) It visualize data flow 
            <br>
            2) smooth concepts flow
        </p>
        """
    st.markdown(intro_description, unsafe_allow_html=True)






# other code

pages_names_to_func = {
    "Introduction": intro,
    "Perceptron Trick": perceptron_trick_main,
    "ANN": ann_main
}











# ---------------- Sidebar -------------- #

st.sidebar.image(logo_url, use_container_width=True)
st.sidebar.markdown(ml_recaper_intro, unsafe_allow_html=True)
st.sidebar.write("")
page_name = st.sidebar.selectbox(
    "Choose a topic",
    pages_names_to_func.keys()
)










# --------------- calling functions ------------- #
global_code()
pages_names_to_func[page_name]() 


