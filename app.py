
import os
import json
import datetime
import gc
import streamlit as st
from streamlit_chat import message
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
import spacy
from googlesearch import search
import queue
import psutil
import numpy as np
import uuid
from datasets import Dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
nltk.download('punkt')
# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to check online status
def online():
    url = "http://www.google.com"
    timeout = 5
    try:
        requests.get(url, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

# Define the base model and tokenizer for summarization
base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Streamlit page configuration
st.set_page_config(page_title="Smart Query Assistant", layout="wide")

# JsonDB Class
class JsonDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.session_folder = "sessions"
        if not os.path.exists(self.session_folder):
            os.makedirs(self.session_folder)

    def store_query(self, session_name, query, answer, timestamp, unique_id):
        embedding = self.model.encode(query, convert_to_tensor=True).tolist()
        session_file = os.path.join(self.session_folder, f"{session_name}.json")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session_data = json.load(f)
        else:
            session_data = {"conversations": []}

        session_data["conversations"].append({
            "unique_id": unique_id,
            "query": query,
            "query_embedding": embedding,
            "answer": answer,
            "timestamp": timestamp.isoformat()
        })

        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=4)

    def retrieve_query(self, query):
        new_embedding = self.model.encode([query], convert_to_tensor=True)
        best_match = None
        best_similarity = -1

        for filename in os.listdir(self.session_folder):
            if filename.endswith(".json"):
                session_name = filename.split(".json")[0]
                with open(os.path.join(self.session_folder, filename), "r") as f:
                    session_data = json.load(f)

                for item in session_data["conversations"]:
                    existing_embedding = np.array(item["query_embedding"]).astype(np.float32)
                    similarity = util.cos_sim(new_embedding, existing_embedding).item()
                    timestamp = datetime.datetime.fromisoformat(item["timestamp"])

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            "session": session_name,
                            "answer": item["answer"],
                            "timestamp": timestamp,
                            "similarity": similarity
                        }

        if best_match and best_similarity > 0.7 and (datetime.datetime.now() - best_match["timestamp"]).days < 30:
            return best_match
        return None

    def update_query(self, session_name, query, answer, timestamp):
        session_file = os.path.join(self.session_folder, f"{session_name}.json")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session_data = json.load(f)

            for item in session_data["conversations"]:
                if item["query"] == query:
                    item["answer"] = answer
                    item["timestamp"] = timestamp.isoformat()

            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=4)

# Session Class
class Session:
    def __init__(self, name):
        self.name = name
        self.conversations = []
        self.offline_queries = queue.Queue()

    def add_conversation(self, query, answer, unique_id):
        self.conversations.append({
            "unique_id": unique_id,
            "query": query,
            "answer": answer,
            "timestamp": datetime.datetime.now().isoformat()
        })

    def add_offline_query(self, query, unique_id):
        self.offline_queries.put({
            "unique_id": unique_id,
            "query": query,
            "timestamp": datetime.datetime.now().isoformat()
        })

# Model1 Class
class Model1:
    def __init__(self, json_db):
        self.json_db = json_db
        self.offline_queries = []

    def search_query(self, query, online=True):
        result = self.json_db.retrieve_query(query)
        if result:
            return result['answer'], result['similarity']
        else:
            if not online:
                self.offline_queries.append(query)
            return None, None

# Model2 Class
class Model2:
    def __init__(self, json_db):
        self.json_db = json_db
        self.original_params = {
            'max_length': 1600,
            'min_length': 1300,
            'do_sample': False,
            'early_stopping': True
        }
        self.summarizer = self.summarization_pipeline(self.original_params)
        self.reward = 0

    @st.cache_resource
    def summarization_pipeline(_self, params):
        return pipeline('summarization', model=base_model, tokenizer=tokenizer, **params)

    def handle_query(self, query, k=10):
        top_k_pages_info = self.get_top_k_webpages_info(query, k)
        summaries = []
        for page in top_k_pages_info:
            summary = self.process_answer(page['text'])
            summaries.append(f"Source of Info: {page['url']}\nInformation: {summary}")
        combined_summary = "\n\n".join(summaries)
        return combined_summary, top_k_pages_info

    def filter_urls(self, urls):
        filtered_urls = []
        for url in urls:
            if any(keyword in url for keyword in [".jpg", ".png", ".apng", ".avif", ".gif", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".svg", ".webp", "youtube.com", "video", "shopping", "book", "flight", "map"]):
                continue
            html = self.fetch_webpage_data(url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                if soup.find_all('p'):
                    filtered_urls.append(url)
        return filtered_urls

    def get_top_k_webpages_info(self, query, k):
        urls = search(query, stop=int(k))
        filtered_urls = self.filter_urls(urls)
        page_data = []
        doc_query = nlp(query)
        query_keywords = [token.lemma_.lower() for token in doc_query if token.is_alpha and not token.is_stop]

        for i, url in enumerate(filtered_urls):
            html = self.fetch_webpage_data(url)
            if html:
                raw_text = self.extract_text_from_html(html)
                text = self.is_related_content(raw_text, query_keywords)
                if text:
                    priority = int(k) - i
                    page_data.append({
                        "priority": priority - 1,
                        "url": url,
                        "text": text,
                    })
        return page_data

    def fetch_webpage_data(self, url):
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                return response.text
            else:
                return None
        except requests.RequestException:
            return None

    def extract_text_from_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(["script", "style", "iframe", "header", "footer", "nav"]):
            script.extract()
        for element in soup.find_all(class_=["ad", "advertisement"]):
            element.extract()
        for element in soup.find_all(id=["ad", "advertisement"]):
            element.extract()
        text = soup.get_text(separator=' ')
        return ' '.join(text.split())

    def is_related_content(self, text, query_keywords):
        doc = nlp(text)
        relevant_text = []
        for sentence in doc.sents:
            sentence_keywords = [token.lemma_.lower() for token in sentence if token.is_alpha and not token.is_stop]
            if any(keyword in sentence_keywords for keyword in query_keywords):
                relevant_text.append(sentence.text.replace('\n', ' '))
        return ' '.join(relevant_text)

    def process_answer(self, context):
        tokens = tokenizer(context, return_tensors='pt', truncation=True, padding=True, max_length=512)
        chunks = [tokens.input_ids[i:i + 512] for i in range(0, len(tokens.input_ids), 512)]
        summaries = []
        for chunk in chunks:
            chunk_text = tokenizer.decode(chunk[0], skip_special_tokens=True)
            summary = self.summarizer(chunk_text, max_length=200, min_length=100, early_stopping=True)
            summaries.append(summary[0]['summary_text'])
        return ' '.join(summaries)

    def update_database(self, session_name, query, combined_summary, sources):
        combined_answer = f"{combined_summary} Sources: " + ", ".join([source['url'] for source in sources])
        self.json_db.update_query(session_name, query, combined_answer, datetime.datetime.now())

    def adjust_parameters(self, feedback):
        if feedback == "like":
            self.reward += 1
        elif feedback == "dislike":
            self.reward = 0
        elif feedback == "regenerate":
            self.reward -= 1

        if self.reward < 0:
            new_params = {
                'max_length': 1600,
                'min_length': 1300,
                'do_sample': True,
                'early_stopping': True,
                'temperature': 0.9,
                'top_k': 50,
                'top_p': 0.95
            }
            self.summarizer = self.summarization_pipeline(new_params)
        else:
            self.summarizer = self.summarization_pipeline(self.original_params)

    def regenerate_response(self, query, k=10):
        self.adjust_parameters("regenerate")
        combined_summary, sources = self.handle_query(query, k)
        self.summarizer = self.summarization_pipeline(self.original_params)
        return combined_summary, sources

# ReinforcementLearning Class
class ReinforcementLearning:
    def __init__(self, model2):
        self.model2 = model2
        self.feedback_data = []

    def collect_feedback(self, feedback):
        self.feedback_data.append(feedback)
        self.model2.adjust_parameters(feedback)


class MainProcessWorkflow:
    def __init__(self, json_db, model1, model2, rl):
        self.json_db = json_db
        self.model1 = model1
        self.model2 = model2
        self.rl = rl
        self.sessions = {}
        self.current_session = None
        self.session_folder = "sessions"
        if not os.path.exists(self.session_folder):
            os.makedirs(self.session_folder)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def start_new_session(self, session_name):
        self.current_session = Session(session_name)
        self.sessions[session_name] = self.current_session
        greeting = "AI: Hello, how may I assist you today?"
        self.current_session.add_conversation("", greeting, "greeting")
        st.session_state.current_session = session_name
        st.session_state.messages = [{"content": greeting, "is_user": False}]

    def handle_query(self, query, k=10, online=online()):
        if self.current_session is None or len(self.current_session.conversations) >= 100:
            session_name = f"Session {len(self.sessions) + 1}"
            self.start_new_session(session_name)

        unique_id = f"{self.current_session.name}_{len(self.current_session.conversations)}"

        if online:
            result = self.json_db.retrieve_query(query)
            if result and result['similarity'] == 1:
                self.json_db.update_query(result['session'], query, result['answer'], datetime.datetime.now())
                return result['answer'], None
            else:
                combined_summary, sources = self.model2.handle_query(query, k)
                self.json_db.store_query(self.current_session.name, query, combined_summary, datetime.datetime.now(), unique_id)
                self.current_session.add_conversation(query, combined_summary, unique_id)
                return combined_summary, sources
        else:
            result, similarity = self.offline_cosine_similarity(query)
            if result:
                self.current_session.add_conversation(query, result, unique_id)
                return result, None
            else:
                self.current_session.add_offline_query(query, unique_id)
                return "Query stored for later processing when online.", None

    def offline_cosine_similarity(self, query):
        query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=True)
        best_match = None
        best_similarity = -1

        for session in self.sessions.values():
            for convo in session.conversations:
                convo_embedding = self.sentence_transformer.encode([convo['query']], convert_to_tensor=True)
                similarity = util.cos_sim(query_embedding, convo_embedding).item()

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = convo

        if best_match and best_similarity > 0.7:
            return best_match['answer'], best_similarity
        return None, None

    def process_offline_queries(self):
        for session in self.sessions.values():
            while not session.offline_queries.empty():
                offline_query = session.offline_queries.get()
                query = offline_query["query"]
                unique_id = offline_query["unique_id"]
                combined_summary, sources = self.model2.handle_query(query)
                self.json_db.store_query(session.name, query, combined_summary, datetime.datetime.now(), unique_id)
                for convo in session.conversations:
                    if convo["unique_id"] == unique_id:
                        convo["answer"] = combined_summary
                        convo["timestamp"] = datetime.datetime.now().isoformat()

    def save_session_to_json(self, session_name):
        session_data = {
            "conversations": self.sessions[session_name].conversations
        }
        filepath = os.path.join(self.session_folder, f"{session_name}.json")
        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=4)

    def load_session_from_json(self, session_name):
        filepath = os.path.join(self.session_folder, f"{session_name}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                session_data = json.load(f)
            session = Session(session_name)
            session.conversations = session_data["conversations"]
            self.sessions[session_name] = session

    def load_all_sessions(self):
        for filename in os.listdir(self.session_folder):
            if filename.endswith(".json"):
                session_name = filename.split(".json")[0]
                self.load_session_from_json(session_name)

    def clear_memory(self):
        gc.collect()

    def reinitialize_models(self):
        self.model1 = Model1(self.json_db)
        self.model2 = Model2(self.json_db)

def handle_user_input(session_id):
    user_input = st.session_state[f"user_input_{session_id}"]
    if user_input:
        mem_before = psutil.Process().memory_info().rss
        st.session_state.messages.append({"content": user_input, "is_user": True})
        message(user_input, is_user=True, key=f"user_message_{len(st.session_state.messages)}")
        result, sources = st.session_state.main_process.handle_query(user_input, 10, online())
        mem_while = psutil.Process().memory_info().rss
        st.write(f"Memory used: {(mem_while) / 1024 / 1024:.2f} MB")
        st.session_state.messages.append({"content": result, "is_user": False})
        message(result, is_user=False, key=f"ai_message_{len(st.session_state.messages)}")
        # Display feedback buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 Like"):
                st.session_state.main_process.rl.collect_feedback("like")
        with col2:
            if st.button("👎 Dislike"):
                st.session_state.main_process.rl.collect_feedback("dislike")
        with col3:
            if st.button("🔄 Regenerate"):
                st.session_state.main_process.rl.collect_feedback("regenerate")
                regenerated_result, _ = st.session_state.main_process.model2.regenerate_response(user_input)
                st.session_state.messages.append({"content": regenerated_result, "is_user": False})
                message(regenerated_result, is_user=False, key=f"ai_message_regenerated_{len(st.session_state.messages)}")

        # Store the query and response in the session file
        st.session_state.main_process.json_db.store_query(session_id, user_input, result, datetime.datetime.now(), f"{session_id}_{len(st.session_state.messages)}")

        # Free up resources
        st.session_state.main_process.clear_memory()
        st.session_state.main_process.reinitialize_models()
        st.session_state["user_input"] = ""
        mem_after = psutil.Process().memory_info().rss
        st.write(f"Memory used: {(mem_after-mem_before) / 1024 / 1024:.2f} MB")
        save_metrics(user_input, result, mem_before, mem_while, mem_after)
        del result, sources
        gc.collect()

# Function to save metrics
def save_metrics(query, response, mem_before, mem_while, mem_after):
    # Tokenize the response (hypothesis) and reference
    reference = [word_tokenize(response)]  # Assuming response is the reference for simplicity
    candidate = word_tokenize(response)

    # Calculate BLEU score
    bleu_score = sentence_bleu(reference, candidate)

    # Calculate ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(response, response)  # Assuming response is the reference for simplicity

    # Calculate METEOR score
    meteor = meteor_score(reference, candidate)

    metrics_data = {
        "query": query,
        "response": response,
        "memory_before": mem_before / 1024 / 1024,
        "memory_during": mem_while / 1024 / 1024,
        "memory_after": mem_after / 1024 / 1024,
        "model_used": "Model1" if "model1" in response else "Model2",
        "timestamp": datetime.datetime.now().isoformat(),
        "bleu_score": bleu_score,
        "rouge_scores": {key: value.fmeasure for key, value in rouge_scores.items()},
        "meteor_score": meteor
    }

    metrics_file = "metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            existing_metrics = json.load(f)
    else:
        existing_metrics = []

    existing_metrics.append(metrics_data)
    with open(metrics_file, "w") as f:
        json.dump(existing_metrics, f, indent=4)


def run_initial_tests():
    test_queries = [
        "What is the capital of France?",
        "Who is the president of the USA?",
        "Summarize the latest news on AI.",
        "What is the capital of France?",
        "Who is the president of the USA?",
        "Summarize the latest news on AI.",
    ]
    metrics = []
    for query in test_queries:
        mem_before = psutil.Process().memory_info().rss
        result, sources = st.session_state.main_process.handle_query(query, 10, online())
        mem_after = psutil.Process().memory_info().rss
        save_metrics(query, result, mem_before, mem_before, mem_after)
        metrics.append({
            "query": query,
            "response": result,
            "memory_before": mem_before / 1024 / 1024,
            "memory_after": mem_after / 1024 / 1024,
            "model_used": "Model1" if "model1" in result else "Model2",
            "timestamp": datetime.datetime.now().isoformat()
        })
    with open("testing.json", "w") as f:
        json.dump(metrics, f, indent=4)

# Main function
def main():
    st.title("Smart Query Assistant")
    json_db = JsonDB()
    model1 = Model1(json_db)
    model2 = Model2(json_db)
    rl = ReinforcementLearning(model2)
    main_process = MainProcessWorkflow(json_db, model1, model2, rl)
    main_process.load_all_sessions()

    # Store main_process in session state for access in handle_user_input
    st.session_state.main_process = main_process

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = None
    if "temp_session_id" not in st.session_state:
        st.session_state.temp_session_id = str(uuid.uuid4())

    # Sidebar for session management
    st.sidebar.title("Conversations")
    if st.sidebar.button("New Conversation"):
        st.session_state.temp_session_id = str(uuid.uuid4())
        st.session_state.messages = []

    # Display existing sessions
    for session_name in main_process.sessions:
        if st.sidebar.button(session_name):
            st.session_state.current_session = session_name
            st.session_state.messages = []
            for convo in main_process.sessions[session_name].conversations:
                st.session_state.messages.append({"content": f"You: {convo.get('query', '')}", "is_user": True})
                st.session_state.messages.append({"content": f"AI: {convo.get('answer', '')}", "is_user": False})

    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=msg["is_user"], key=f"message_{i}")

    # Input area at the bottom
    session_id = st.session_state.current_session or st.session_state.temp_session_id
    st.text_input("Enter your query:", key=f"user_input_{session_id}", on_change=lambda: handle_user_input(session_id))

    st.markdown("""
    <style>
    .stTextInput > div > div > input {
        background-color: #000000;
        color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 20px;
        padding: 10px 15px;
        border: none;
    }
    .stButton > button {
        border-radius: 20px;
        padding: 5px 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Process offline queries if we're back online
    if online():
        main_process.process_offline_queries()
    # Run initial tests and save metrics
    if not os.path.exists("testing.json"):
        run_initial_tests()

if __name__ == "__main__":
    main()
