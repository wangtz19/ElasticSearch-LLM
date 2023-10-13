# ElasticSearch LLM 实现思路 
> 王通泽    2023-10-11
## 1 数据处理
已有政策文件全为`.txt`格式，存储路径为`es-llm/data/cleaned_data_all`，多数文件具备较好层次结构，例如
```
中华人民共和国主席令第五十五号

　　《中华人民共和国合伙企业法》已由中华人民共和国第十届全国人民代表大会常务委员会第二十三次会议于2006年8月27日修订通过，现将修订后的《中华人民共和国合伙企业法》公布，自2007年6月1日起施行。

　　中华人民共和国主席　胡锦涛

　　2006年8月27日
...
第一章 总则
第一条 为了规范合伙企业的行为，保护合伙企业及其合伙人、债权人的合法权益，维护社会经济秩序，促进社会主义市场经济的发展，制定本法
第二条 本法所称合伙企业，是指自然人、法人和其他组织依照本法在中国境内设立的普通合伙企业和有限合伙企业
...
第十四条 设立合伙企业，应当具备下列条件
  （一）有二个以上合伙人。合伙人为自然人的，应当具有完全民事行为能力；    （二）有书面合伙协议；    （三）有合伙人认缴或者实际缴付的出资；    （四）有合伙企业的名称和生产经营场所；    （五）法律、行政法规规定的其他条件。
...
```
当前处理思路为：
1. 使用正则抓取文本目录，并以此切分文本。为了避免切分后的文本过短，目前仅抓取了`第x章`，`第x条`和`x、`（其中x表示汉字数字一、二、三...），可以根据测试结果适当调整目录抓取的范围，代码参见`es-llm/src/es/text_splitter.py`
2. 上述文档可以为如下格式
```
{
	"简介": "《中华人民共和国合伙企业法》已由中华人民共和国第十届全国人民代表大会常务委员会第二十三次会议于2006年8月27日修订通过，现将修订后的《中华人民共和国合伙企业法》公布，自2007年6月1日起施行。
　　中华人民共和国主席　胡锦涛
　　2006年8月27日...",
　　"第一章 总则 第一条 为了规范合伙企业的行为，保护合伙企业及其合伙人、债权人的合法权益，维护社会经济秩序，促进社会主义市场经济的发展，制定本法": "",
　　"第一章 总则 第十四条 设立合伙企业，应当具备下列条件": "（一）有二个以上合伙人。合伙人为自然人的，应当具有完全民事行为能力；    （二）有书面合伙协议；    （三）有合伙人认缴或者实际缴付的出资；    （四）有合伙企业的名称和生产经营场所；    （五）法律、行政法规规定的其他条件。"
}
```
3. 对于名为`title`的文件在第2步生成的字典`data`，将其每个`{key: val}`项单独组织为
```
{
	"标题": title,
	"子标题": key,
	"内容": val
}
```
也就是说，一个文件会生成若干个`{"标题": title, "子标题": key, "内容": val}`的项，将所有文件的项合并为列表`doc_list`，代码参见`es-llm/src/es/text_loader.py`
4. 将`doc_list`用于构建`es index`（在es中，index表示数据库），每个文本段的索引字段即为`标题`、`子标题`和`内容`。创建`index`时，设置生成索引的默认`analyzer`为`ik_max_word`（多种方式分词）,设置检索的默认`analyzer`为`ik_smart`（仅一种分词方式）；添加索引字段`vector`，其类型为`densen_vector`，用于后续向量检索（实现向量检索时可以根据需要修改）。代码参见`es-llm/src/es/create_index.py`
## 2 检索方法
### 字符检索
状态：已完成
目前仅根据`标题`、`子标题`和`内容`三个字段进行检索，匹配方法为`multi_match`，建议尝试更多方法（例如`query_string`，参考[Query DSL | Elasticsearch Guide [8.10] | Elastic](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html)），代码参见`es-llm/src/es/my_elasticsearch.py`
### 字符检索 + 向量筛选
状态：已完成，待测试及优化
对于一个`query`，通过字符检索得到相似度最高的`es-top-k`个文本段之后，利用`text2vec`计算`query`与每个文本段的相似度，最终得到`vec-top=k`个文本段（`es-top-k`大于`vec-top-k`），代码参见`es-llm/src/chat/chat_model.py`
### es向量检索
状态：待完成
思路：根据构建索引时生成`vector`字段（或者根据需求另外新建字段），调用[k-nearest neighbor (kNN) search](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)或者[Semantic search](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-search.html)
## 3 运行方法
### 运行 es 后端
目前`ik`的最新版本均为`8.9.0`，`es`需要与之适配，同样为`8.9.0`。由于es不能用root账户运行，目前通过`tmux`将其运行在名为`es`的用户下
```
tmux # 或者进入已有session, tmux attach-session -t <session-id>
su es
cd elasticsearch-8.9.0/bin
./elasticsearch
```
### 创建 es index
```
cd es-llm/src/es
python create_index.py
```
### 运行 es llm
1. web demo
```
cd es-llm
CUDA_VISIBLE_DEVICES=0 python web_demo.py
```
2. api demo
```
cd es-llm
CUDA_VISIBLE_DEVICES=0 python api_demo.py
```
## 4 后续方向
1. 字符检索 + 向量筛选测试，以及`text2vec`向量计算相似度优化（目前很慢）
2. es 向量检索，参考[k-nearest neighbor (kNN) search | Elasticsearch Guide [8.10] | Elastic](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)和[Semantic search | Elasticsearch Guide [8.10] | Elastic](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-search.html)
3. 对比不同检索方法的效果，参考[Query DSL | Elasticsearch Guide [8.10] | Elastic](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html)
4. 探索不同相似度计算方法的效果，参考[Similarity module | Elasticsearch Guide [8.10] | Elastic](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html)
5. ik分词器字典扩展，参考[Elasticsearch（ES）分词器的那些事儿 - 简书 (jianshu.com)](https://www.jianshu.com/p/021393d29fc4)