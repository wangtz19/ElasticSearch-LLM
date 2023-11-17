实现向量检索，knn召回
实现多路召回


# ElasticSearch LLM

## A QA Bot based on `elasticsearch`、`text2vec` and `LLM(ChatGLM2 etc)`

### Run
##### 1. Install Dependency
`pip install -r requirements.txt`
##### 2. Prepare ElasticSearch
First of all, you are recommended to install and es locally, [here](https://www.elastic.co/guide/en/elasticsearch/reference/8.10/targz.html) is a detailed official guide. For Chinese text retrieval, the [ik pluggin](https://github.com/medcl/elasticsearch-analysis-ik) is a prerequisite for `Analyzer` and `Tokenizer`. In order to simplify development, you cound disable authorization and SSL by modify `config/elatiscsearch.yml` under your es installation directory. Note that, for security consideration, es is not allowed to run under the root user.

Suppose you run es in the default address and port (i.e. `127.0.0.1:9200`), execute following instructions to create your es index for local files (only support `.txt` files currently)
```
cd src/es
python create_index.py
```

#### 3. Chat with LLM
1.  For API demo
```
python api_demo.py [-h] [--host HOST] [--port PORT] [--filepath FILEPATH] [--model_name MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --host HOST
  --port PORT
  --filepath FILEPATH   path to the local knowledge file
  --model_name MODEL_NAME
```
2.  For Web demo
```
python web_demo.py [-h] [--model_name MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME, -m MODEL_NAME
```

<h5>
  Reference
  <br/>
  https://github.com/imClumsyPanda/langchain-ChatGLM
  <br/>
  https://github.com/shibing624/text2vec
  <br/>
  https://github.com/iGangao/es_text2vec_chatglm_qa
</h5>
