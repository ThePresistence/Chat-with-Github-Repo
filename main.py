import subprocess
import os
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai
from dotenv import load_dotenv

load_dotenv()



def create_database(github_link): 
    process = subprocess.Popen(['python', 'src/main.py', 'process', '--repo-url', github_link])
    process.wait()
    print('done')

    dataset_name = github_link.split('/')[-1]

    return dataset_name 


def search_db(db, query):
    """Search for a response to the query in the DeepLake database."""
    # Create a retriever from the DeepLake instance
    retriever = db.as_retriever()
    # Set the search parameters for the retriever
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10
    # Create a ChatOpenAI model instance
    model = ChatOpenAI(model="gpt-3.5-turbo")
    # Create a RetrievalQA instance from the model and retriever
    qa = RetrievalQA.from_llm(model, retriever=retriever)
    # Return the result of the query
    return qa.run(query)

def ask_multiple_question_and_save_it_in_text_file(dataset_name):
    inputs_questions = ["What is the aim of the project? ", 
                        "Where is the data resoruces is comming from? If possible mention the stack like azure or aws?", 
                        "Create a digram, that represents the workflow of the project?", 
                        "Explain Insights about the project ? ", 
                        "Additional information for developers ?", 
                        "Explain how deployment is done and which stack is used like azure or aws? "]
    
    activeloop_username = os.environ.get("ACTIVELOOP_USERNAME")
    activeloop_dataset_path = (
        f"hub://{activeloop_username}/{dataset_name}"
    )


    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Create an instance of OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Create an instance of DeepLake with the specified dataset path and embeddings
    db = DeepLake(
        dataset_path=activeloop_dataset_path,
        read_only=True,
        embedding_function=embeddings,
    )


    all_outputs = []

    with open("all_output.txt", 'w') as file: 
        for i in inputs_questions: 
            user_input = i 
            output = search_db(db, user_input)
            print(output)
            all_outputs.append(output)
            file.write(output)
  

# database_name = create_database("https://github.com/rpgeeganage/alls")
# out = ask_multiple_question_and_save_it_in_text_file(database_name)

import os, openai, langchain

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM


# !pip install openai
import os
from openai import OpenAI

gpt_4_key = "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"
os.environ["OPENAI_API_KEY"] = gpt_4_key
client = OpenAI()

## Funciton for read me maker
def give_answer_readme(content):
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "Pretend you are README file creator for github. I have provided the aim of the project, data resources, workflow diagram, Insights about the project, deployment and insights for the developers. Take a detailed note of everything and create a markdown file out of it. That markdown file needs to understand by the developers, write the content in very easy terms and understandable. I need the ouput in markdown file"},
      {"role": "user", "content": content}
    ]
  )

  return completion.choices[0].message.content


def give_answer_transcription_video(content):

  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You're a video creator expert in explaining the project to developers. User will give you the aim, workflow diagram, deployment details, data resrouces, insights of the project and additional information of the project. By using this information create for 2 min. Explain all the information properly."},
      {"role": "user", "content": content}
    ]
  )

  return completion.choices[0].message.content


with open("all_output.txt", 'w') as file: 
    file.write("ohhh")