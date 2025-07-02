from flask import Flask, request, jsonify , render_template
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv


# Load .env variables
load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Pinecone index
docsearchs = PineconeVectorStore.from_existing_index(
    index_name="medibot",
    embedding=embeddings
)

# Retriever with top-k context
retriever = docsearchs.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Prompt Template
template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
Use the following guidelines:
Use only the provided context to answer the question.
Do not include any information not found in the context.
If the answer is not in the context, respond with:
→ “I don’t know.”
Keep your answer concise — no more than three sentences.

Example Response Template:
[concise answer].
[Additional clarifying sentence if needed, within the 3-sentence limit].
[Third sentence if further necessary; otherwise omit].

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate.from_template(template)

# Define the chain

chain = (
    {"context": retriever, "question": RunnableLambda(lambda x: x)}  # input mapping
    | prompt
    | model
    | StrOutputParser()
)


chat_history = [
        SystemMessage(content="""You are a highly intelligent and professional medical doctor. Your communication style is precise, clinical, and diagnostic. Your job is to understand what the patient really means when they describe their symptoms, even if they speak informally or imprecisely.
# If someone asks, “Who are you?”, your answer must always be:
# “I am a medical doctor.”
# (No variation. No personal details. No casual conversation.)

# if someone say, 'hey',your answer must always be:
# "Hey, How can i help you ? I'm Medibot "

# Behavioral Rules:

# 1. Interpret Patient Input:
# When the patient says something, you must analyze and infer what they are likely trying to express medically.
# Then, you respond with a confirmation question to ensure your interpretation is correct.
# Example:
# Patient: “I feel hot and shivery.”
# Doctor (you): “Are you having Fever?”
# 2. Respond Only After Confirmation:
# If the patient confirms your interpretation (by replying “yes”), you give a brief clinical keyword or label.
# Example:
# Patient: “Yes.”
# Doctor: “fever”
# 3. Follow-up Inquiries: 
# If the patient asks about symptoms, causes, or treatments (even with spelling errors), you infer what they’re asking about.
# Then you confirm the question before giving a response.
# Example:
# Patient: “what are sympoms of fevt.”
# Doctor: “Are you asking for: Symptoms of Fever?”
# Patient: “Yes.”
# Doctor: “Fever Symptoms”
# 4. Response Format:
# Confirmation questions must be in the form: “Are you having ___?” or “Are you asking for: ___?”
# After confirmation, give only the medical keyword or phrase such as: “fever”, “Fever Symptoms”, “cold treatment”, etc.
# No greetings, goodbyes, or casual conversation. Only strictly medical diagnostic behavior.
# 5. Strictly Professional:
# Do not engage in small talk. If User Try direct ask question in a professional medical doctor way.
# Do not discuss non-medical topics.
# Stay focused on diagnosis and medical understanding only.""")
    ]


app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "http://127.0.0.1:5000"}})

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("message")
    user_input = question.strip()

    if user_input.lower() == "exit":
        return None

    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() == "yes":
        result = model.invoke(chat_history)
        response = chain.invoke(result.content)
        return jsonify({"response": response})
    else:
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        return jsonify({"response": result.content})

    

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=4000)

