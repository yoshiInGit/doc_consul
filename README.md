﻿# Langchainを使った社内資料コンシェルジェ（開発中）

## 開発背景
生成AIの発展が盛んな今の時代、よりこの技術を実際の業務やビジネスに生かせるようになる必要性を感じました。
そこでLangChainを利用し、より実践的、応用的な生成AIの活用に挑戦しました。

「社内資料のコンシェルジェ」というテーマで簡易的なRAG機能とAI機能を組み合わせた、シンプルなプロジェクトです。

### 今回作るもの：社内文書のコンシェルジュ

今回作成してみるサービスの題材として、「社内文書のコンシェルジュ」を設定してみます。
イメージとしては、AIが社内文書の保管されているフォルダを参照し、こちらの質問に応じて資料の解説をしてくれるものです。
保存されている資料が膨大で複雑なもののであるとき、そもそもこの資料が自分の求めているのか調べるだけで体力を使ってしまいますが、そこをAIに助けてもらおうといものです。

### 具体的なイメージ:
- あなたは、社内文書の保管フォルダにアクセスする権限を持つAIアシスタントに話しかけます。
- 「○○の資料について教えて」と指示します。
- AIアシスタントは、フォルダ内の文書を分析し、当該資料の解説を行って


## プロジェクト構成

#### 入力
```py
print("どの資料を読み込みますか？")
file_path = input("ファイルパスを入力してくだい: ")
```

#### 資料を数値化
```py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

os.environ["GOOGLE_API_KEY"] = "APIキー"

loader = PyPDFLoader(file_path)

pages = loader.load_and_split()

# 資料の数値化のためのEmbeddingを用意
embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 資料を数値化し、Chromaに保存
vectorstore = Chroma.from_documents(pages, embedding=embed)

# 資料を数値化したものを検索可能な状態にする
retriever = vectorstore.as_retriever()
```

#### AIとの対話開始
```py
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
```

#### クエリに基づいて関連性の高いドキュメントを検索
```py
## クエリに基づいて関連性の高いドキュメントを検索
relevant_documents = retriever.get_relevant_documents(query)

# 一つの文字列にまとめる
combined_document = ""
for doc in relevant_documents:
    combined_document += doc.page_content + "\n\n"  # 各ドキュメントの間に改行を2つ入れる

print("\n\n関連するドキュメント：")
print(combined_document)
```

#### 参考資料を使って応答する
```py
prompt2 = PromptTemplate.from_template("""
### 参考資料
{combined_document}
                                       
### 質問
{query}
                                       
### 指示
あなたの目的は、与えられた資料をもとに、質問に答えることです。
ただし、あなたはコンシェルジュのようにふるまってください。
それでは、質問に答えてください。
コンシェルジェ：
                                    """)

chain = prompt2 | llm

print("\n\n入力されたプロンプト：")
print(prompt2.format(combined_document=combined_document, query=query))

print("\n\nコンシェルジェ：")
print(chain.invoke({"combined_document" : combined_document, "query" : query}).content)
```

#### 結果
```
どの資料を読み込みますか？
ファイルパスを入力してくだい: ./docs/sample.pdf


コンシェルジェ：
かしこまりました。お客様、本日はどのようなご質問でしょうか？資料の内容について、どんな些細なことでも構いませんので、お気軽にお尋ねください。お客様の ご要望に沿えるよう、精一杯お手伝いさせていただきます。
入力：AIの倫理的課題について教えて


関連するドキュメント：
⽣成 AI 最新動向レポート
2024 年 10 ⽉時点の情報
⽬次
(中略)
6.3 ビジネスモデルの変⾰
AIaaS （ AI as a Service ）: 業界特化型の AI サービス提供が⼀般化し、中⼩企業でも⾼度な AI 活⽤が
可能になっています。

入力されたプロンプト：

### 参考資料
⽣成 AI 最新動向レポート
2024 年 10 ⽉時点の情報
⽬次
1. 概要
（中略）
6.3 ビジネスモデルの変⾰
AIaaS （ AI as a Service ）: 業界特化型の AI サービス提供が⼀般化し、中⼩企業でも⾼度な AI 活⽤が
可能になっています。



### 質問
AIの倫理的課題について教えて

### 指示
あなたの目的は、与えられた資料をもとに、質問に答えることです。
ただし、あなたはコンシェルジュのようにふるまってください。
それでは、質問に答えてください。


ンシェルジェ：
はい、喜んでお答えします。AIの倫理的課題についてですね。

資料によりますと、主に以下の3つの課題が挙げられています。

1.  **コンテンツ真正性:** AIが生成する偽情報やディープフェイクの問題です。見破ることが難しい精巧な偽物が作られることで、社会的な混乱や誤解を招く可能 性があります。この問題に対して、検出技術の開発や規制による対応が進められています。

2.  **著作権問題:** AIの学習データや生成されたコンテンツの著作権に関する議論が活発になっています。AIが学習したデータに著作権がある場合、生成されたも のにも影響があるのか、誰に権利があるのかなどが課題となっています。

3.  **バイアスと公平性:** AIモデルに内在するバイアスの問題です。学習データに偏りがあると、AIが特定のグループに対して不公平な判断を下す可能性がありま す。バイアスの検出と緩和が重要な課題となっています。

これらの課題に対して、国際的な規制フレームワークの策定や、業界による自主的な取り組みが進められています。例えば、AIシステムの判断根拠の説明可能性を高 める技術開発や、AI生成コンテンツの識別を可能にするデジタル署名技術の導入などが挙げられます。

何かさらに詳しく知りたい点や、別の角度からの質問などございましたら、お気軽にお申し付けください。
```

## 今後の展望
機能面における中核的な実装はできました。
今後はより利用可能な状態になるよう
- UIの実装
- 文書検索の充実
- 会話履歴の有効化
に挑戦していきたいです。
