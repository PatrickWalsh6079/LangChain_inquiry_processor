import re

import openai
import streamlit as st
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
# from vllm import LLM, SamplingParams
from labels import candidate_labels, candidate_sents

load_dotenv()


def get_response(inquiry):
    # text-davinci-003
    # gpt-3.5-turbo
    # togethercomputer/RedPajama-INCITE-Base-3B-v1
    if st.session_state['conversation'] is None:
        llm = OpenAI(
            temperature=0,
            model_name='text-davinci-003'
        )

        st.session_state['conversation'] = ConversationChain(
            llm=llm,
            verbose=True
        )

    prompt = f"""
                    You are a helpful AI assistant that helps to process customer inquiries. For each inquiry, provide a response.

                    Inquiry:
                    "{inquiry}"
                    
                    In the response include:
                    1. Key issue of this inquiry.
                    2. Reduce the key issue to 5 words or less.
                    3. The inquiry sentiment.
                    
                    Use this response template:
                    Key issue: '1'.\n
                    Summary: '2'.\n
                    Tone: '3'.
    """


    # facebook/bart-large-mnli
    # alexandrainst/scandi-nli-large
    # facebook/opt-iml-max-30b  <-- memory times out, model too large
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")
    ###
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # outputs = classifier.generate(prompt, sampling_params)
    # # Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ###

    results = classifier(inquiry, candidate_labels, multi_label=True)
    count = 0
    threshold = 0.80

    #     print(text)
    label_results = []
    # zero shot
    for i in results['scores']:
        if i < threshold:
            break
        #         print(results['labels'][count], i)
        label_results.append(results['labels'][count] + str(i))
        count += 1
    # label_results = [' {0.}'.format(elem) for elem in label_results]
    formatted_labels = []
    for item in label_results:
        item = re.sub("0[.]", ' 0.', item)
        formatted_labels.append(item)
    print(formatted_labels)
    label_results = '\n\n'.join(formatted_labels)

    # sentiment
    template = "The sentiment of this inquiry is {}"
    predictions = classifier(inquiry,
                             candidate_sents,
                             multi_label=True,
                             hypothesis_template=template
                             )

    response = st.session_state['conversation'].predict(input=prompt)
    print(st.session_state['messages'])

    sent_labels = predictions['labels']
    sent_scores = predictions['scores']
    predictions = str(sent_labels) + str(sent_scores)
    response = response + '\n\n\nIssue Codes:\n\n' + label_results + '\n\n\nSentiment:\n\n' + predictions

    return response
