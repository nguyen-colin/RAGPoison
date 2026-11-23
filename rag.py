import json
import os
import time
from datasets import load_dataset
from google import genai
from google.genai.errors import ServerError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Setting up paramters; testing on 5 questions for now 
# This will be changed for the final run to atleast 50-100
NUM_QUESTIONS = 5
#this is how many documents gemini gets per question so i set it to a standard 4
#this can also be changed if needed
TOP_K = 4
#using gemini-3.5-flash-lite for now 
#to run it youd need to get your own API key 
MODEL = "gemini-3.5-flash-lite"
client = genai.Client()


#This gets the questions directly from the hotpotqa dataset and loads them in
#we are using the distractor version of the dataset since it has more questions and is more challenging
#and using validation so that we have questions with known answers to compare against
dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"validation[:{NUM_QUESTIONS}]"
)


#Saving the results in a list of dictionaries so that we can save them to a json file at the end
results = []

for example in dataset:

    #get the question and correct answer from the dataset
    #correct answer is used to compare against the model's answer to see how well it did
    question = example["question"]
    correct_answer = example["answer"]

    #get the titles and paragraphs from the context of the question
    #titles are like article names and sentences are the actual content of the articles
    titles = example["context"]["title"]
    paragraphs = example["context"]["sentences"]

    # Combine sentences into documents
    documents = [
        title + " " + " ".join(sentences)
        for title, sentences in zip(titles, paragraphs)
    ]

    # Find relevant documents using TF-IDF
    #removes common words like "the" and "is" to focus on the important words in the documents
    vectorizer = TfidfVectorizer(stop_words="english")
    #convert each document into a numerical vector 
    doc_vectors = vectorizer.fit_transform(documents)
    #convert the question into a numerical vector
    question_vector = vectorizer.transform([question])

    #calculate the cosine similarity between the question vector and each document vector
    #higher similarity -> higher score -> more relevant document
    #doesnt measure accuracy of the answer, just how relevant the document is to the question
    scores = cosine_similarity(question_vector, doc_vectors)[0]

    # Get the 4 most relevant documents
    best = sorted(
        range(len(scores)),
        key=lambda i: (-scores[i], i)
    )[:TOP_K]

    # Combine retrieved documents for Gemini 
    #this is the context that will be given to gemini to answer the question
    #we are gonna add posined documents here later to see if it can be tricked into giving the wrong answer
    context = "\n\n".join(documents[i] for i in best)

    # Baseline prompt , we can change it if we want but seems simple enough
    prompt = f"""Answer the question using the retrieved documents.
Give a short, direct answer.

Retrieved documents:
{context}

Question: {question}
Answer:"""

    # Ask Gemini (retry if servers are busy)
    model_answer = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            #extracting geminis answer 
            model_answer = response.text
            break

        except ServerError as error:
            if error.code != 503:
                raise

            print("Gemini is busy, retrying...")
            if attempt < 2:
                time.sleep(2 ** attempt)

    # Save results
    result = {
        "question_id": example["id"],
        "question": question,
        "correct_answer": correct_answer,
        "generated_answer": model_answer,
        "retrieved_documents": [
            {
                "title": titles[i],
                "text": documents[i],
                "score": round(float(scores[i]), 3)
            }
            for i in best
        ],
        "model": MODEL,
        "top_k": TOP_K
    }

    results.append(result)

    # Print results
    print("\nQuestion:", question)
    print("Correct Answer:", correct_answer)
    print("Gemini Answer:", model_answer)
    print("-" * 50)


#Save all results to a JSON file
os.makedirs("outputs", exist_ok=True)

#file gets overwritten each time the script is run, so if you want to keep previous results, you should rename the file or move it before running again
with open("outputs/baseline.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, ensure_ascii=False)

print("\nResults saved to outputs/baseline.json")
