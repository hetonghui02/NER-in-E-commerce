from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import pandas as pd

df = pd.read_excel("product_name.xlsx")

item_name = df["item_name"].tolist()

brand_name_list = []
model_name_list = []
category_name_list = []


ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_chinese-base-ecom-50cls')


for name in item_name:
    result = ner_pipeline(name)
    output = result['output']
    brand_spans = list(set([item['span'] for item in output if item['type'] == '品牌']))
    model_spans = list(set([item['span'] for item in output if item['type'] == '款式_其他']))
    category_spans = list(set([item['span'] for item in output if item['type'] == '产品_核心产品']))
    brand_name_list.append(brand_spans)
    model_name_list.append(model_spans)
    category_name_list.append(category_spans)

df["brand_name"] = brand_name_list
df["model_name"] = model_name_list
df["category_name"] = category_name_list

df.to_excel("name_classification.xlsx",index=False)

'''
result = ner_pipeline('小雨越南尾货已过检过验流浪包系列流浪包单肩斜挎手提背包香奶奶流浪背囊小号双肩包双肩链条包女包小书包')
output = result['output']
brand_spans = [item['span'] for item in output if item['type'] == '品牌']
model_spans = [item['span'] for item in output if item['type'] == '款式_其他']
category_spans = [item['span'] for item in output if item['type'] == '产品_核心产品']
print("brand_name",brand_spans)
print("model_name",model_spans)
print("category_name",category_spans)
'''
# {'output': [{'type': '品牌', 'start': 0, 'end': 2, 'span': 'eh'}, {'type': '品牌', 'start': 3, 'end': 6, 'span': '摇滚狗'}, {'type': '款式_其他', 'start': 6, 'end': 8, 'span': '涂鸦'}, {'type': '款式_其他', 'start': 8, 'end': 10, 'span': '拔印'}, {'type': '款式_其他', 'start': 10, 'end': 12, 'span': '宽松'}, {'type': '材质_面料', 'start': 12, 'end': 14, 'span': '牛仔'}, {'type': '产品_核心产品', 'start': 14, 'end': 15, 'span': '裤'}, {'type': '款式_其他', 'start': 16, 'end': 19, 'span': '情侣款'}]}