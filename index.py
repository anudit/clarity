import gradio as gr
import validators
from dotenv import load_dotenv
from langchain.cache import InMemoryCache
from langchain.chains import (LLMChain, MapReduceDocumentsChain,
                              ReduceDocumentsChain, StuffDocumentsChain)
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader, YoutubeLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from getlinks import depth_lookup

load_dotenv()
set_llm_cache(InMemoryCache())

llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')
embeddings = OpenAIEmbeddings()
html2text = Html2TextTransformer(ignore_images=True, ignore_links=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3072, chunk_overlap = 100)

global vectorstore
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def import_website(url, depth=3):

    if validators.url(url):

        links = depth_lookup(url, depth)

        loader = AsyncChromiumLoader(links)
        docs = loader.load()
        docs = html2text.transform_documents(docs)
        vectorstore.add_documents(docs)

    else:
        gr.Warning('Invalid Url: ' + url )

    return len(vectorstore.get()['documents']) if vectorstore.get()['documents'] else 0

def import_page(url):

    if validators.url(url):

        loader = AsyncChromiumLoader([url])
        docs = loader.load()
        docs = html2text.transform_documents(docs)
        vectorstore.add_documents(docs)
    
    else:
        gr.Warning('Invalid Url: ' + url )

    return len(vectorstore.get()['documents']) if vectorstore.get()['documents'] else 0

def predict(message, history):

    context = vectorstore.similarity_search_with_relevance_scores(message)

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], 
        template="""
        CONTEXT:{context}

        {chat_history}
        Human: {human_input}
        Chatbot:"""
    )

    llm_chain = LLMChain(
        prompt=prompt, 
        llm=llm, 
        verbose=True, 
        memory=ConversationBufferMemory(k=3, memory_key="chat_history", input_key="chat_history")
    )
    output = llm_chain.run(human_input=message, context=context[:2])

    return output

def transcribe_video(url):
    
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=True
    )
    docs = loader.load()

    vectorstore.add_documents(docs)
    return docs[0].page_content, len(vectorstore.get()['documents']) if vectorstore.get()['documents'] else 0

def summarize(url, map_template, reduce_template):

    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=True
    )
    docs = loader.load()
    splits = text_splitter.split_documents(docs)

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    return map_reduce_chain.run(splits)

with gr.Blocks(title="Clarity") as demo:

    gr.Markdown(
        """
        # Clarity
        Understand anything about anything
        """
    )

    with gr.Tabs():
        with gr.TabItem(label="Chat") as t1:

            with gr.Row():

                docs_count = gr.Label(value=len(vectorstore.get()['documents']), label="Docs in Databse")

                with gr.Column():

                    input_url = gr.Textbox(label="URL", lines=1)
                    
                    depth = gr.Slider(2, 5, value=3, step=1, label="Search Depth")
                    
                    with gr.Row():
                        btn = gr.Button(value="Import Website")
                        btn.click(import_website, inputs=[input_url, depth], outputs=[docs_count])        
                        
                        btn2 = gr.Button(value="Import Page")
                        btn2.click(import_page, inputs=[input_url], outputs=[docs_count])        
                    

            gr.ChatInterface(predict)

        with gr.TabItem(label="Podcast Chat") as t1:

            gr.Markdown(""" ### Chat and Summarize Podcasts """)

            with gr.Column():

                with gr.Row():
                    transcript = gr.TextArea(label="Transcript", lines=20) 

                    with gr.Group():
                        with gr.Row():
                            map_template = gr.TextArea(label="Map Prompt Template", value="""The following is a set of documents
                                {docs}
                                Based on this list of docs, please identify the main themes 
                                Helpful Answer:""", lines=3)
                            reduce_template = gr.TextArea(label="Reduce Prompt Template", value="""The following is set of summaries:
                                {doc_summaries}
                                Take these and distill it into a final, consolidated summary of the main themes. 
                                Helpful Answer:""", lines=3)

                        summary = gr.TextArea(label="Summary ~(10s/10m)", lines=15)

                with gr.Group():
                    video_url = gr.Textbox(label="Youtube URL", lines=1, value="https://www.youtube.com/watch?v=UZtQcnt12SM")
                    
                    with gr.Row():

                        btn = gr.Button(value="Import Video")
                        btn.click(transcribe_video, inputs=[video_url], outputs=[transcript, docs_count])
                        
                        btn2 = gr.Button(value="Summarize")
                        btn2.click(summarize, inputs=[video_url, map_template, reduce_template], outputs=[summary])
                

if __name__ == "__main__":
    demo.queue().launch(server_port=8000)
