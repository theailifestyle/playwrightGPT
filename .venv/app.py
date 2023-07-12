import os
from apikey import api_key
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import CTransformers

os.environ['OPENAI_API_KEY'] = api_key

#app framework
st.title('🦜🔗 Playwright GPT')

scene=st.text_input('Enter a scene:')
char1_bio=st.text_input('Enter a character bio 1:')
char2_bio=st.text_input('Enter a character bio 2:')
dialogue_lines = st.number_input('Enter number of dialogue lines:', min_value=0)
plot_twist_line = st.number_input('After how many lines should the plot twist occur?:', min_value=0)
plot_twist=st.text_input('Enter a Plot twist:')
char2_start=st.text_input('Enter the first line for the second character:')

apikey=api_key
dialogue=""

start_button = st.button('Start Dialogue')

# Prompt templates
llm = OpenAI(openai_api_key =apikey, temperature=.7)
char1_template = PromptTemplate(input_variables = ['char1_biop','dial_hist1'], template='you are the first character in a play You are {char1_biop}. This is the what was said till now: {dial_hist1} \\n respond with the next dialogue as you would in a play.Only respond with your character\'s dialogue \\n')
char1_chain = LLMChain(llm=llm, prompt=char1_template, output_key='char1line')

llm2 = OpenAI(openai_api_key =apikey, temperature=.7)
char2_template = PromptTemplate(input_variables = ['char2_biop','dial_hist2','char1line'],template='you are the second character in a play You are {char2_biop}. This is the what was said till now:{dial_hist2} \\n {char1line} \\n respond with the next dialogue as you would in a play. Only respond with your character\'s dialogue \\n')
char2_chain = LLMChain(llm=llm2, prompt=char2_template, output_key='char2line')

dialogue_chain = SequentialChain(chains=[char1_chain, char2_chain],input_variables=['char1_biop','char2_biop','dial_hist1','dial_hist2'],output_variables=['char1line','char2line'],verbose=True)

if start_button:
    char1_name = char1_bio.split()[0]  # Get the first word from the bio
    char2_name = char2_bio.split()[0]  # Get the first word from the bio
    dialogue += f"Narrator: The scene is set in {scene}. {char1_bio} and {char2_bio} are in a play.\\n {char2_name}: {char2_start}\\n"
    st.write(dialogue)
    for i in range(dialogue_lines):
        if i == plot_twist_line:
            dialogue += f"Narrator: {plot_twist}\n"
            st.write(f"Narrator: {plot_twist}")
        response=dialogue_chain({'char1_biop':char1_bio,'char2_biop':char2_bio,'dial_hist1':dialogue,'dial_hist2':dialogue})
        st.write(response['char1line'])
        dialogue += f"{response['char1line']}\n"
        st.write(response['char2line'])
        dialogue += f"{response['char2line']}\n"

    llm3 = OpenAI(openai_api_key =apikey, temperature=.7)
    title_template = PromptTemplate(input_variables = ['dialoguesummary'], template='Come up with an exciting title for this play: \\n {dialoguesummary} \\n')
    title_chain = LLMChain(llm=llm3, prompt=title_template, output_key='PlayTitle')
    title = title_chain.run(dialogue)
    st.title(title)


