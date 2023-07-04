# NER-in-E-commerce
some NER ways in E-commerce area
# 商品标题信息提取方法的调研和比较

## 商品标题的信息提取或分类可以使用自然语言处理（NLP）技术来解决，主要可使用命名实体识别（NER）的方式来识别出商品标题中属于商品品牌、商品颜色或代号以及商品种类类别的命名实体，并通过识别后的命名实体进行分类和标注。以下是三个需要提取的内容以及相关的算法和方法的概述。

### 基于规则的方法：
1. 使用预定义的品牌列表和规则来匹配商品标题中的品牌名称。这种方法适用于已知品牌数量有限的情况。
2. 使用正则表达式或关键词匹配等方法，提取商品标题中的型号、代号和颜色信息。这种方法对于特定格式的商品标题可能比较有效，但可能无法处理变化较大的文本。
3. 构建一个商品类目的关键词表，通过匹配商品标题中的关键词来确定商品的类目或种类。这种方法简单直接，但需要维护一个准确的关键词表。

### 基于序列标注的方法：
1. 序列标注使用隐马尔可夫模型（HMM）
2. 可以使用命名实体识别（Named Entity Recognition，NER）的方法，将商品标题中的型号、代号和颜色作为实体进行识别和提取。常用的序列标注模型包括条件随机场（CRF）和循环神经网络（RNN）等。

### 基于神经网络和机器学习的方法：
1. 可以使用文本分类算法，如朴素贝叶斯、支持向量机（SVM）或深度学习模型，来训练一个品牌分类器，将商品标题分类为不同的品牌类别。
2. 可以使用文本分类算法，如朴素贝叶斯、支持向量机（SVM）或深度学习模型，来训练一个商品类目分类器，将商品标题分类为不同的类目。


## 具体方法和运行结果展示：
### 基于贝叶斯分类的分类器，将商品标题分类为不同类别
#### 训练过程
1. 数据准备
2. 数据预处理
3. 特征提取，使用词袋模型进行特征提取并将标题由文本表示转换为向量表示
4. 创建品牌、型号、种类编码器和分类模型

#### 分类结果展示
![image](https://github.com/hetonghui02/NER-in-E-commerce/assets/36235543/33a903e3-776a-47dc-beb7-b4df487db99d)



### 序列标注方法：HMM方法和CRF方法
#### 标记精确度、召回率和混淆矩阵
1. HMM模型：![image](https://github.com/hetonghui02/NER-in-E-commerce/assets/36235543/41d06637-98b5-4784-98d6-02d42a72fcba)


2. CRF模型：

#### 预测结果
1. HMM模型：![image](https://github.com/hetonghui02/NER-in-E-commerce/assets/36235543/8449806d-a51c-4330-9068-e4dfae6ca5c7)

2. CRF模型：![image](https://github.com/hetonghui02/NER-in-E-commerce/assets/36235543/b9121d5f-28f3-45dd-8de1-67109da89bfd)


###机器学习方法：

#### 标记精确度和混淆矩阵：
1. Bilstm模型：![image](https://github.com/hetonghui02/NER-in-E-commerce/assets/36235543/ccb0a488-dfe3-4ab0-88e5-0618acbf58f7)

2.Bilstm+CRF模型：![image](https://github.com/hetonghui02/NER-in-E-commerce/assets/36235543/c07053a7-a129-476a-b8d2-bcf8aa3a6fb5)

####  预测结果
1. Bilstm模型：![image](https://github.com/hetonghui02/NER-in-E-commerce/assets/36235543/bc59a975-5ba1-4790-b8e0-9498985231df)

2.Bilstm-CRF模型：![image](https://github.com/hetonghui02/NER-in-E-commerce/assets/36235543/e4020878-7e24-4cfd-8f41-ca8f4d93e45a)


### model scope模型：RaNER
采用Transformer-CRF模型，使用StructBERT作为预训练模型底座，结合使用外部工具召回的相关句子作为额外上下文，使用Multi-view Training方式进行训练。

#### 模型介绍：
Transformer-CRF模型是一种结合了Transformer和条件随机场（CRF）的模型，用于命名实体识别任务。Transformer是一种基于自注意力机制的神经网络结构，能够有效地捕捉句子中的上下文信息。CRF则用于建模标签之间的依赖关系，以提高标签序列的一致性。

StructBERT是一种基于BERT（Bidirectional Encoder Representations from Transformers）的模型，通过引入结构化信息和预训练任务来提升预训练模型的性能。它在BERT的基础上增加了一个结构化嵌入层，用于编码句子中的语法和结构信息。

使用外部工具召回相关句子作为额外上下文可以增加句子的语境信息，有助于提升模型的性能。这些召回的句子可能是通过搜索引擎或其他方法获取的与原始句子相关的文本。将这些召回的句子与原始句子进行拼接构建多视图输入，可以为模型提供更全面的上下文信息。

Multi-view Training方式是一种训练策略，它通过鼓励不同视图的相似性来提高模型的性能。在这种方法中，通过共享参数和共同的损失函数，同时训练基于原始句子的输入视图和基于召回的输入视图，使它们产生相似的上下文表示或输出标签分布。

这种方法结合了Transformer-CRF模型、StructBERT预训练模型、外部工具召回的相关句子和Multi-view Training方式，旨在通过引入额外的上下文信息和多视图训练来提高命名实体识别模型的性能。

#### 预测分类结果：
![image](https://github.com/hetonghui02/NER-in-E-commerce/assets/36235543/98b20eb9-fe95-4146-b30a-90fd4c17e48a)

