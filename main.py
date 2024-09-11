from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack.components.builders import PromptBuilder
# from haystack.nodes import PreProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import requests
import json
from bs4 import BeautifulSoup
import re
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
import spacy


headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
    }


def find_movie_url_by_title(title):
    query = title.replace(' ', '+')
    search_url = f'https://search.douban.com/movie/subject_search?search_text={query}&cat=1002' # cat=1002 for movie

    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if 'type' in script.attrs and script.attrs['type'] == 'text/javascript':
                if 'window.__DATA__' in script.text:  
                    data_script = script.text
                    break

        match = re.search(r'window\.__DATA__ = ({.*?});', data_script, re.DOTALL)
        if match:
            json_data = match.group(1)
            try:
                data = json.loads(json_data)
                movie_ids = [item['id'] for item in data['items']]
            except json.JSONDecodeError as e:
                print("JSONDecodeError:", e)

    return movie_ids


def find_review_by_movie_id(movie_ids, num=0):
    movie_id = movie_ids[num]
    review_url = f'https://movie.douban.com/subject/{movie_id}/'
    response = requests.get(review_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        review_lists = soup.find('div', class_='review-list')
        review_data = []
        for review_item in review_lists.find_all('div', class_='review-item')[0:1]:    ## only fetch the first review
            review_item_feedback = _fetch_review_data(review_item)   
            review_data.append(review_item_feedback)  
        # print(len(review_data))    
        return review_data


def _fetch_review_data(review_item, translate='zh2en'):
    ## fetch meta data
    review_id = review_item.get('id')
    
    date = review_item.find('span', class_='main-meta').text.strip()
    useful_count = review_item.find('span', id=lambda x: x and x.startswith('r-useful_count-')).text.strip()
    rating_class = review_item.find('span', class_=lambda x: x and x.startswith('allstar'))
    if rating_class:
        rating = rating_class['class'][0] 
        star_rating = rating.replace('allstar', '')
        star_rating = {int(star_rating) / 10} 
    else:
        star_rating = "No rating"

    print(f"Review_id: {review_id}")
    print(f"Date: {date}")
    print(f"Useful Count: {useful_count}")
    print(f"Rating: {star_rating}")
    print("---------------")
    
    ## fetch review content
    review_content = _fetch_review_content(review_id)
    if translate == 'zh2en':
        review_content = _translate_zh2en(review_content)
    
    ## save the review content and its translation to a file
    filename = f"{movie_name}-{review_id}.txt"
    with open(filename, "a") as f:
        f.write(review_content)

    ## create a dictionary to store the data
    review_item_feedback = {
        'review_id': review_id,
        'date': date,
        'useful_count': useful_count,
        'rating': star_rating,
        'content': review_content
    }

    return review_item_feedback
    

def _fetch_review_content(review_id):

    url = f'https://movie.douban.com/review/{review_id}/'
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
    
        link_report_id = f'link-report-{review_id}'
        review_content_div = soup.find('div', id=link_report_id)
        
        if review_content_div:
            review_text = review_content_div.get_text(separator='\n', strip=True)
            return review_text
        else:
            return "Failed to retrieve the page."


def _translate_zh2en(src_text):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

    sentences = _split_into_sentences(src_text)
    translations = ''

    for sentence in sentences:
        translated = tokenizer.prepare_seq2seq_batch([sentence], return_tensors="pt")
        output = model.generate(**translated)
        translation = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        translations += translation

    return translations


def _split_into_sentences(text):
    nlp = spacy.load("zh_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def review_to_document(review_data):
    documents = [
        Document(
            content=review['content'],
            meta={
                'review_id': int(review['review_id']),
                'date': review['date'],
                'useful_count': int(review['useful_count']),
                'rating': review['rating']
            }
        ) for review in review_data
    ]

    # Create an instance of InMemoryDocumentStore
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()

    docs_with_embeddings = doc_embedder.run(documents)
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs_with_embeddings["documents"])

    return document_store




def piperr(document_store):
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = InMemoryEmbeddingRetriever(document_store)
    prompt_builder = DocumentJoiner()

    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)
    
    generator = LlamaCppGenerator(model="/Users/yeyous/rr_code/LangChain_demo/openchat-3.5-1210.Q3_K_S.gguf", n_ctx=0, n_batch=128)
    generator.warm_up()

    basic_rag_pipeline = Pipeline()
    # Add components to your pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)

    # Now, connect the components to each other
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    # basic_rag_pipeline.connect("retriever", "llm")
    basic_rag_pipeline.connect("prompt_builder", "llm")
    return basic_rag_pipeline


def run(document_store, questions):

    basic_rag_pipeline = piperr(document_store)

    for question in questions:
        response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
        print(response["llm"]["replies"][0])

if __name__ == "__main__":
    movie_name = "triangle"              ## name of the movie
    num = 0                             ## most relevant item 
    question = ["who is the director?"]   ## question list to ask

    movie_ids = find_movie_url_by_title(movie_name)
    review_data = find_review_by_movie_id(movie_ids, num)
    document_store = review_to_document(review_data)
    run(document_store, question)