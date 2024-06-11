# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import gradio as gr
import time
# from functools import lru_cache
from llama_cpp import Llama
import pandas as pd
import argparse


# Argument parser
parser = argparse.ArgumentParser(description="Vietcombank Chatbot")
parser.add_argument('--model_path', type=str, default="./data/model/mistral-7b-openorca.Q4_0.gguf", help='Path to the model file')
parser.add_argument('--embedding_path', type=str, default='all-MiniLM-L6-v2.gguf2.f16.gguf', help='Path to the embedding model file')
parser.add_argument('--vector_db_path', type=str, default="./data/dbv3", help='Path to the vector database file')
parser.add_argument('--csv_path', type=str, default='./data/questions_answers.csv', help='Path to the CSV file containing questions and answers')
parser.add_argument('--debug', type=int, default=0, help='Turn on debug mode (1) or off (0)')
parser.add_argument('--history', type=int, default=0, help='Turn on history feature (1) or off (0)')

args = parser.parse_args()

model_path = args.model_path
embedding_path = args.embedding_path
vector_db_path = args.vector_db_path

# model_path = "./model/vinallama-7b-chat_q5_0.gguf"


# Load model and database once at startup
# llm = CTransformers(
#     model=model_path,
#     model_type="llama",
#     temperature=0.25,
#     streaming=True
# )
llm = Llama(model_path=model_path,
            n_gpu_layers=1, n_ctx=4096)

embedding_model = GPT4AllEmbeddings(
    model_name=embedding_path,
    gpt4all_kwargs={'allow_download': 'True'}
)

def read_vector_db():
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db
df = pd.read_csv(args.csv_path)
db = read_vector_db()

template_context = '''system\nBạn là AI chatbot của Vietcombank. Sử dụng thông tin sau để trả lời câu hỏi của người dùng.\n
{context}\nuser\n{question}\nassistant\n'''
prompt_template = PromptTemplate(template=template_context, input_variables=["context", "question"])

def add_to_vector_db(db, prompt, response):
    doc = Document(page_content=f"Prompt: {prompt}\nResponse: {response}")
    db.add_documents([doc])

# @lru_cache(maxsize=100)  # Cache the results of the last 100 queries
def get_response_from_chain(db, query):
    doc = db.similarity_search_with_relevance_scores(query, k=3)
    # breakpoint()
    res = []
    for d, score in doc:
        if d.page_content[-1] == '?':
                content = df['answer'][df['question']==d.page_content[:-1]]
                res.append((Document(page_content=content.values[0]), score))
        else:
            res.append((d, score))
    return res

def handle_user_input(message, history):
    # breakpoint()
    if args.debug:
        start_time = time.time()
        results = get_response_from_chain(db, message)
        end_time = time.time()
        print(f"find in db time: {end_time - start_time} seconds")
    
    
        start_time = time.time()
        # context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results if score >= 0.3])
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = prompt_template.format(context=context_text, question=message)
        print(context_text)
        print(f'score: {[score for _doc, score in results]}')
        response_text = ""
        output = llm.create_completion(prompt, max_tokens=500, stop=["<|im_end|>"], stream=True)
        for token in output:
            # breakpoint()
            response_text += token["choices"][0]["text"]
            yield response_text  # Yield each token to display incrementally
        if args.history:
            add_to_vector_db(db, message, response_text)

        end_time = time.time()
        print(f"generate response time: {end_time - start_time} seconds")
    else:
        results = get_response_from_chain(db, message)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = prompt_template.format(context=context_text, question=message)
        response_text = ""
        output = llm.create_completion(prompt, max_tokens=500, stop=["<|im_end|>"], stream=True)
        for token in output:
            response_text += token["choices"][0]["text"]
            yield response_text
        if args.history:
            add_to_vector_db(db, message, response_text)

# Define the Gradio interface
# interface = gr.ChatInterface(fn=handle_user_input)
theme = gr.themes.Default(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="neutral",
).set(
    border_color_accent_subdued='*border_color_accent',
    block_shadow='none',
    block_shadow_dark='none',
    form_gap_width='0px',
    checkbox_label_background_fill='*button_secondary_background_fill',
    checkbox_label_background_fill_dark='*button_secondary_background_fill',
    checkbox_label_background_fill_hover='*button_secondary_background_fill_hover',
    checkbox_label_background_fill_hover_dark='*button_secondary_background_fill_hover',
    checkbox_label_shadow='none',
    error_background_fill_dark='*background_fill_primary',
    input_background_fill='*neutral_100',
    input_background_fill_dark='*neutral_700',
    input_border_width='0px',
    input_border_width_dark='0px',
    input_shadow='none',
    input_shadow_dark='none',
    input_shadow_focus='*input_shadow',
    input_shadow_focus_dark='*input_shadow',
    stat_background_fill='*primary_300',
    stat_background_fill_dark='*primary_500',
    button_shadow='none',
    button_shadow_active='none',
    button_shadow_hover='none',
    button_transition='background-color 0.2s ease',
    button_primary_background_fill='*primary_200',
    button_primary_background_fill_dark='*primary_700',
    button_primary_background_fill_hover='*button_primary_background_fill',
    button_primary_background_fill_hover_dark='*button_primary_background_fill',
    button_primary_border_color_dark='*primary_600',
    button_secondary_background_fill='*neutral_200',
    button_secondary_background_fill_dark='*neutral_600',
    button_secondary_background_fill_hover='*button_secondary_background_fill',
    button_secondary_background_fill_hover_dark='*button_secondary_background_fill',
    button_cancel_background_fill='*button_secondary_background_fill',
    button_cancel_background_fill_dark='*button_secondary_background_fill',
    button_cancel_background_fill_hover='*button_cancel_background_fill',
    button_cancel_background_fill_hover_dark='*button_cancel_background_fill',
    button_cancel_border_color='*button_secondary_border_color',
    button_cancel_border_color_dark='*button_secondary_border_color',
    button_cancel_text_color='*button_secondary_text_color',
    button_cancel_text_color_dark='*button_secondary_text_color'
)
interface = gr.ChatInterface(
    handle_user_input,
    # chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Hỏi tôi câu hỏi về ngân hàng Vietcombank", container=False, scale=7),
    title="Vietcombank's Chatbot",
    # description="Hỏi câu hỏi về ngân hàng Vietcombank và tôi sẽ trả lời cho bạn.",
    # theme="soft",
    theme=theme,
    examples=["Bạn là ai","Tôi có thể liên kết bao nhiêu thẻ VCB trên Ứng dụng MOCA?", "Tôi có nhất thiết phải sử dụng Cookies cho trình duyệt hay không?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()

# Launch the Gradio interface
interface.launch()
