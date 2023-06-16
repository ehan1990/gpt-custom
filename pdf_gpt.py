import os
import sys

from PyPDF2 import PdfReader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

REQUIRED_ENVS = ["OPENAI_API_KEY", "PDF_FILE"]


def check_envs(required_envs):
    for env in required_envs:
        if not os.environ.get(env):
            print(f"{env} not found. exiting...")
            sys.exit(1)


def main():
    check_envs(REQUIRED_ENVS)
    pdf_file = os.environ["PDF_FILE"]
    reader = PdfReader(pdf_file)

    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    print(f"Finished reading {pdf_file}. Now you can ask me questions!\n")
    while True:
        query = input("Question (type exit to quit): ")
        if query == "exit":
            sys.exit(0)
        docs = docsearch.similarity_search(query)
        ans = chain.run(input_documents=docs, question=query)
        print("\n" + ans + "\n")


if __name__ == "__main__":
    main()

