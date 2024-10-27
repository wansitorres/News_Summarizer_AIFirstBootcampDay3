import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer", page_icon="", layout="wide")

# Custom CSS to center the title and the API input
st.markdown("""
    <style>
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    .centered-title {
        font-size: 3rem;
        font-weight: bold;
        color: #333;
    }
    .api-input-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Centered title
st.markdown("""
    <div class="title-container">
        <h1 class="centered-title">News Summarizer</h1>
    </div>
    """, unsafe_allow_html=True)

# Centered API key input
if 'openai_api_key' not in st.session_state:
    st.markdown('<div class="api-input-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        if openai_api_key:
            # Basic check: ensure it starts with "sk-" and has a reasonable length
            if openai_api_key.startswith("sk-") and len(openai_api_key) > 20:
                st.success("API key provided!")
                st.session_state.openai_api_key = openai_api_key
                openai.api_key = openai_api_key
                st.rerun()
            else:
                st.warning("Please enter a valid OpenAI API key.")
        else:
            st.info("Please enter your OpenAI API key to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Main app logic
    with st.sidebar:
        options = option_menu(
            "Menu", 
            ["Home", "About Us", "Model"],
            icons = ['book', 'globe', 'tools'],
            menu_icon = "list", 
            default_index = 0,
            styles = {
                "icon" : {"color" : "#dec960", "font-size" : "20px"},
                "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
                "nav-link-selected" : {"background-color" : "#262730"}
            })

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = None

    if options == "Home":
        st.title("Home Page")
        st.write("Welcome to News Summarizer! Click the Model button in the sidebar to start summarizing news articles.")
    

    elif options == "About Us":
        st.title("About Us")
        st.write("This is a tool that summarizes news articles made by Juan Cesar Torres. This was made as a project for the AI First bootcamp by AI Republic.")

    elif options == "Model":
        st.title("News Summarizer")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            News_Article = st.text_area(
                "Enter News Article Here:",
                placeholder="Paste your news article here...",
                height=300 
            )
            submit_button = st.button("Generate Summary")
        
        if submit_button:
            with st.spinner("Generating Summary..."):
                System_Prompt = """
You are an assistant who excels at summarizing news articles in a clear, concise, and neutral manner. Your objective is to read each article thoroughly and distill the key information into a structured summary, following these specific guidelines:

Headline Summary:

Goal: Create a brief, one-sentence overview that captures the main point or most significant takeaway of the article.
Length: Aim for no more than 20 words.
Tone: Neutral and objective; avoid sensational language or subjective phrasing.
Key Points:

Goal: Identify and list the major facts, events, or findings presented in the article. Focus on the who, what, when, where, and why of the news.
Details to include:
Names of key individuals, organizations, or locations mentioned in the article.
Quantitative details (numbers, statistics, dates) that give context or significance.
Brief explanations or descriptions of any complex ideas, terms, or events as needed for clarity.
Structure: Provide this in 2-4 concise bullet points, summarizing each significant fact or event in about one sentence.
Context and Background:

Goal: Offer essential background information that would help the reader understand the broader significance or relevance of the news.
Details to include:
Relevant historical, political, economic, or social background if it adds to the understanding.
Mention any related events or previous developments that the article connects to.
Length: Limit this section to 1-2 sentences or bullet points, keeping it concise yet informative.
Implications and Possible Outcomes:

Goal: Summarize any potential consequences, follow-up actions, or next steps mentioned in the article, focusing on how the event might impact relevant stakeholders or the public.
Details to include:
Any official statements, planned actions, or anticipated effects that could result from the news.
Mention how the article indicates the news might influence broader trends, policies, or communities.
Structure: Present this in 1-2 bullet points, highlighting only significant potential impacts.
Language and Tone:

Objective: Ensure a factual, objective tone throughout the summary. Avoid inserting personal opinions, exaggeration, or assumptions.
Clarity: Use simple, accessible language, focusing on clarity and readability for a general audience.
Length: Aim for a concise yet complete summary, generally not exceeding 5-7 sentences overall.
When summarizing, prioritize accuracy, clarity, and brevity to provide readers with a well-rounded understanding of the article's most critical information without unnecessary details.
"""
            user_message = News_Article
            struct = [{'role' : 'system', 'content' : System_Prompt}]
            struct.append(  {'role' : 'user', 'content' : user_message})
            chat = openai.ChatCompletion.create(model = 'gpt-4o-mini', messages = struct)
            response = chat.choices[0].message.content
            struct.append({'role' : 'assistant', 'content' : response})
            st.write("Assistant:", response)
