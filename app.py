
import streamlit as st
from streamlit_chat import message
from model import get_response

# set initial page session state
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = ""


# Setting page title and header
st.set_page_config(page_title="AVA Inquiry Processor", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>How can I assist you? </h1>", unsafe_allow_html=True)


st.sidebar.title("Inquiry Results")

response_container = st.container()

# Here we will have a container for user input text box
container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=False):
        inquiry = st.text_area("Insert AVA Inquiry:", key='input', height=250)
        submit_button = st.form_submit_button(label='Process')

        if submit_button:

            # st.session_state['messages'].append(inquiry)
            model_response = get_response(inquiry)
            st.session_state['messages'] = model_response

            with response_container:
                summarise_placeholder = st.sidebar.write(
                    "Results for AVA Inquiry:\n\n", st.session_state['messages'])
            #     for i in range(len(st.session_state['messages'])):
            #         if (i % 2) == 0:
            #             message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
            #         else:
            #             message(st.session_state['messages'][i], key=str(i) + '_AI')
