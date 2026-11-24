import pickle as pkl
from reviews_preprocessing import String_processing
import streamlit as st
import plotly.express as px
import pandas as pd 

read_xlsx = pd.read_excel("data_set/Virtual Assistant Devices Dataset _Final Version.xlsx",usecols=["Sentiment","Aspect(s)"])
sentmient_val = []
sentiment = ["Positive", "Negative", "Neutral"]
for val in sentiment:
    values = read_xlsx[read_xlsx["Sentiment"] == val]
    sentmient_val.append(len(values))
    print(values["Sentiment"].value_counts())
print(sentmient_val)
aspect_common = read_xlsx["Aspect(s)"].value_counts() 
aspect_index = aspect_common[aspect_common > 80].index
aspect_common_val = aspect_common[aspect_common > 80].values
print(aspect_common)
print(aspect_common.dtype)

aspect_enco = pkl.load(open("Models/Aspect_encoder.pkl","rb"))
aspect_model = pkl.load(open("Models/Aspect_model.pkl","rb"))
aspect_vec = pkl.load(open("Models/Aspect_vectorizor.pkl","rb"))
sentiment_enco= pkl.load(open("Models/Sentiment_encoder.pkl","rb"))
sentiment_model = pkl.load(open("Models/Sentiment_model.pkl","rb"))
sentiment_vec = pkl.load(open("Models/Sentiment_vectorizor.pkl","rb"))
print(aspect_model ,aspect_vec , sentiment_model)

def user_input(user_input):
    input_process = String_processing(user_input)
    print(input_process)
    input_aspect_vectorize = aspect_vec.transform([input_process])
    input_sentiment_vectorize = sentiment_vec.transform([input_process])
    print(input_aspect_vectorize.toarray())
    aspect_model_pred = aspect_model.predict(input_aspect_vectorize)
    sentiment_model_pred = sentiment_model.predict(input_sentiment_vectorize)
    return sentiment_enco.inverse_transform([sentiment_model_pred][0]),aspect_enco.inverse_transform([aspect_model_pred][0]) 

# user_inp = input("Drop a Comment : ")
# result = user_input(user_inp)
# print(f" Predictied Aspect: {result[0][0]}\n",f"Predicted Sentiment: {result[1][0]}" )
# Sidebar

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300..800;1,300..800&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
     @import url("https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200"); 


        h1, h2, h3, h4, p, label, button, input, textarea,a {
             font-family: "Poppins", sans-serif !important;
        }
        div[data-testid="stSidebarHeader"] {
            display : none !important ;
        }
        div[data-testid="stSidebarUserContent"] .st-emotion-cache-zh2fnc {
            width: 100% !important;
                }
        div[data-testid="stSidebarUserContent"] button[data-testid="stBaseButton-secondary"] {
            padding: 10px;
            border: none;
            background: none;
        }
        .sidebar-btn:hover{
            background: white;
        }   

        .stHeading{
            text-align: center;
            margin: 15px 0px 30px 0;
        }
        span[data-testid="stIconMaterial"] {
            font-size: 25px !important;
        }
        div[data-testid="stSidebarUserContent"] p{
            font-size: 18px !important;
        }
        div[data-testid="stSidebarUserContent"] div[data-testid="stCaptionContainer"] {
            position: fixed;
            bottom: 10px;
            width: 250px;
        }
        div[data-testid="stCaptionContainer"] p{
                text-align: center;
                font-size: 15px !important;
        }
        .stAppHeader{
            display: none !Important ;
        }
        .stMain h1{
                font-size: 35px !important;
                
                padding: 0.25rem 0px 2rem !important;
        }
        .stMain .stMainBlockContainer{
                padding: 0px !important;
        }
        div[data-baseweb="tab-panel"]{
            border-radius: 10px;
            background: #f0f2f6;
            padding: 10px 40px 20px
        }
            h3 {
                        padding-bottom: 0px !important;
                        margin: 50px 0 0 !important;
                        font-size: 28px;
            }
        .stTabs h3 {
            font-size: 18px !important;
            text-align : center; 
            padding: 25px 0 10px !important;
            margin: 0 !important;
        }
        div[data-baseweb="tab-border"]{
            background-color: white !important; 

        }
        div[data-baseweb="tab-list"]{
                margin-left: 12px;
        }
        .stMainBlockContainer{
            max-width: 85% !important;
        }
        button[data-baseweb="tab"] {
            background: #f0f2f6;
            padding: 12px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        .chart-wrapper{
                margin-top: 45px;
            }
        .sidebar-btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
                border-radius: 0.5rem;
                min-height: 2.5rem;
                margin: 0px;
                line-height: 1.6;
                font-size: 18px !important;
                width: 100%;
                cursor: pointer;
                color: rgb(49 51 63) !important;
                border: none;
                background: none;
                text-decoration: none !important;
                margin: 10px 0;

        }
        html {
                scroll-behavior: smooth;
        }
        .material-symbols-outlined {
                font-variation-settings:
                'FILL' 0,
                'wght' 400,
                'GRAD' 0,
                'opsz' 25,    
        }
        .material-symbols-outlined{
                font-size: 25px !important;
                margin-right: 5px;
            }

            # @media (max-width: 600px){
            # div[data-baseweb="tab-list"] p{
            #         font-size: 10px;
            # }
            # button[data-baseweb="tab"]{
            #     padding: 7px !important;
            # }
            # .stMain h1{
            #     font-size: 25px !important;
            # }
            # .st-emotion-cache-1frkdi4 h3 {
            #     font-size: 20px;
            #     margin: 10px 0 0 !important;
            # }
            </style>
""", unsafe_allow_html=True)


st.sidebar.title("ANALYSIS HUB")

st.sidebar.markdown(f"""
<a href="#prediction-section" class="sidebar-btn">
  <span class="material-symbols-outlined">
neurology
</span> Prediction Models
</a>
""", unsafe_allow_html=True)


st.sidebar.markdown(f"""
<a href="#sentiment-section" class="sidebar-btn">
    <span class="material-symbols-outlined">
analytics
</span> Sentiment Analysis
</a>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<a href="#aspect-section" class="sidebar-btn">
  <span class="material-symbols-outlined">search_insights</span> Aspects Analysis
</a>
""", unsafe_allow_html=True)

st.sidebar.caption("A PEL Internship Project")

st.markdown('<div id="prediction-section"></div>', unsafe_allow_html=True)
st.title("Product Research & Intelligence System")
# Tabs
tab1, tab2 = st.tabs([
    "💬 Single Comment Analysis",
    "📁 Full File Analysis"
])


# --- TAB 1: SINGLE COMMENT ANALYSIS ---
with tab1:

    st.markdown("<h3>One-to-One Analysis</h3>", unsafe_allow_html=True)
    st.caption("Analyze an individual customer comment.")

    with st.form("single_comment_form"):

        users_input = st.text_area(
            "Customer Comment",
            placeholder="Enter a customer review here...",
            height=150
        )

        submitted_comment = st.form_submit_button(
            "Analyze Comment",
            use_container_width=True
        )

        if submitted_comment:
            if users_input.strip():
                Model_prediction = user_input(users_input)
                result_df = pd.DataFrame({
                "Prediction Type": ["User Comment", "Sentiment","Aspect"],
                "Prediction": [users_input, Model_prediction[0][0],Model_prediction[1][0]]
                })
                st.dataframe(result_df, hide_index=True)
        else:
                st.warning("Please enter a customer comment.")


# --- TAB 2: FULL FILE ANALYSIS ---
with tab2:

    st.markdown("<h3>Full File Analysis</h3>", unsafe_allow_html=True)
    st.caption("Upload a dataset to analyze multiple customer comments.")
    with st.form("full_file_form"):

        uploaded_file = st.file_uploader(
            "Upload Customer Reviews",
            type=["csv", "xlsx"]
        )

        submitted_file = st.form_submit_button(
            "Analyze File",
            use_container_width=True
        )

        if submitted_file:

            if uploaded_file is not None:

                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file)
                    else :
                        st.warning("Please upload a valid file format (CSV or Excel).")
                    st.success("File uploaded successfully!")
                    data_file = df["Comments"].apply(lambda x : user_input(x))
                    st.write("Comment Prediction & Aspect Analysis")
                    st.dataframe(data_file, use_container_width=True)

                    st.write(f"Total Records: {len(df)}")

                    # Connect your sentiment and aspect prediction models here.
                    # Example:
                    # df["Sentiment"] = sentiment_model.predict(df["Comments"])
                    # df["Aspect"] = aspect_model.predict(df["Comments"])

                except Exception as e:
                    st.error(f"Unable to read file: {e}")

            else:
                st.warning("Please upload a file first.")

with st.container(): 
    st.markdown('<div id="sentiment-section"></div>', unsafe_allow_html=True)
    st.subheader("Customer Review Sentiment Distribution")
    data = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [sentmient_val[0],sentmient_val[1], sentmient_val[2]]
    })
    fig = px.pie(
        data,
        names="Sentiment",
        values="Count",
        hole=0
    )
    st.plotly_chart(fig, use_container_width=True)

data = pd.DataFrame({
    "Aspect": [aspect_index[0], aspect_index[1], aspect_index[2], aspect_index[3], aspect_index[4],aspect_index[5],aspect_index[6]],
    "Count": [aspect_common_val[0], aspect_common_val[1], aspect_common_val[2], aspect_common_val[3], aspect_common_val[4],aspect_common_val[5],aspect_common_val[6]]
})


st.markdown('<div id="aspect-section"></div>', unsafe_allow_html=True)
st.subheader("Customer Review Aspect Distribution")

st.bar_chart(
    data,
    x="Aspect",
    y="Count",
    color="#4F46E5"
)