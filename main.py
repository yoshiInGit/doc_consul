## ファイルの読み込み
print("どの資料を読み込みますか？")
file_path = input("ファイルパスを入力してくだい: ")


## ファイルの数値化
import os
import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for GOOGLE_API_KEY: ")

loader = PyPDFLoader(file_path)

pages = loader.load_and_split()

# 資料の数値化のためのEmbeddingを用意
embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 資料を数値化し、Chromaに保存
vectorstore = Chroma.from_documents(pages, embedding=embed)

# 資料を数値化したものを検索可能な状態にする
retriever = vectorstore.as_retriever()


## AIとの対話開始
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt1 = PromptTemplate.from_template("""
あなたの目的は、与えられた資料をもとに、質問に答えることです。
ただし、あなたはコンシェルジュのようにふるまってください。
現在資料は与えられているという前提で対話をすすめてください。
それでは、資料のどのようなことについて知りたいか聞いてください。
コンシェルジェ：
                                    """)

chain = prompt1 | llm

print("\n\nコンシェルジェ：")
print(chain.invoke({}).content)
query = input("入力：")


## クエリに基づいて関連性の高いドキュメントを検索
relevant_documents = retriever.get_relevant_documents(query)

# 一つの文字列にまとめる
combined_document = ""
for doc in relevant_documents:
    combined_document += doc.page_content + "\n\n"  # 各ドキュメントの間に改行を2つ入れると区切りが見やすいです

print("\n\n関連するドキュメント：")
print(combined_document)


## 参考資料を使って応答する
prompt2 = PromptTemplate.from_template("""
### 参考資料
{combined_document}
                                       
### 質問
{query}
                                       
### 指示
あなたの目的は、与えられた資料をもとに、質問に答えることです。
ただし、あなたはコンシェルジュのようにふるまってください。
それでは、質問に答えてください。
                                    """)

chain = prompt2 | llm

print("\n\n入力されたプロンプト：")
print(prompt2.format(combined_document=combined_document, query=query))

print("\n\nコンシェルジェ：")
print(chain.invoke({"combined_document" : combined_document, "query" : query}).content)







