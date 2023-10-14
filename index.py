import gradio as gr
import validators
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader, YoutubeLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from getlinks import depth_lookup

load_dotenv()

llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')
embeddings = OpenAIEmbeddings()
html2text = Html2TextTransformer(ignore_images=True, ignore_links=True)

global vectorstore
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

global docs_count
global transcript

def import_website(url):

    if validators.url(url):

        links = depth_lookup(url, 3)

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
        memory=ConversationBufferMemory(k=2, memory_key="chat_history", input_key="chat_history")
    )
    output = llm_chain.run(human_input=message, context=context[:2])

    return output

def transcribe_video(url):
    
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=True
    )
    docs = loader.load_and_split();
    vectorstore.add_documents(docs)
    return docs, len(vectorstore.get()['documents']) if vectorstore.get()['documents'] else 0

def summarize(trans):
    return "hello"

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

                docs_count = gr.Label(value=len(vectorstore.get()['documents']), label="Docs in Databse", scale=1)

                with gr.Group():

                    input_url = gr.Textbox(label="URL", lines=1,scale=4)
                    
                    with gr.Row():
                        btn = gr.Button(value="Import Website")
                        btn.click(import_website, inputs=[input_url], outputs=[docs_count])        
                        
                        btn2 = gr.Button(value="Import Page")
                        btn2.click(import_page, inputs=[input_url], outputs=[docs_count])        
                    

            gr.ChatInterface(predict)

        with gr.TabItem(label="Podcast Chat") as t1:

            gr.Markdown(
                """
                ### Chat and Summarize Podcasts
                """
            )

            with gr.Column():

                with gr.Row():
                    transcript = gr.TextArea(label="Transcript", lines=10) 

                    with gr.Group():
                        summary = gr.TextArea(label="Summary", lines=10)
                        btn = gr.Button(value="Summarize")
                        btn.click(summarize, inputs=[transcript], outputs=[summary])

                
                with gr.Group():
                    video_url = gr.Textbox(label="Youtube URL", lines=1, value="https://www.youtube.com/watch?v=UZtQcnt12SM")
                    
                    btn = gr.Button(value="Import Video")
                    btn.click(transcribe_video, inputs=[video_url], outputs=[transcript, docs_count])

        

if __name__ == "__main__":
    demo.queue().launch(server_port=8000)
