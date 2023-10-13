import os

import gradio as gr
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.memory import ConversationBufferMemory

from getlinks import depth_lookup

os.environ["OPENAI_API_KEY"] = "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"

global docs_count

llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')
memory = ConversationBufferMemory(memory_key="chat_history")

def build_database(url):

    links = depth_lookup(url, 3)

    loader = AsyncHtmlLoader(links)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs = html2text.transform_documents(docs)

    return len(docs)


def predict(message, history):

    prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template="""
    {chat_history}
    Human: {human_input}
    Chatbot:""")

    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, memory=memory)
    output = llm_chain.predict(human_input=message)

    return output

with gr.Blocks() as demo:
    
    gr.Markdown(
        """
        # Clarity
        Understand anything about anything
        """
    )

    with gr.Row():

        docs_count = gr.Label(value=0, label="Docs in Databse")

        with gr.Group():

            input_url = gr.Textbox(label="URL", lines=1, scale=4)
            
            btn = gr.Button(value="Import Website", scale=1)
            btn.click(build_database, inputs=[input_url], outputs=[docs_count])        


    gr.ChatInterface(predict)

if __name__ == "__main__":
    demo.queue().launch()
