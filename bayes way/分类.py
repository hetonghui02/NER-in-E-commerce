import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# 1. 数据准备
data = pd.read_excel("product_name.xlsx")  # 加载标题和标签信息的CSV文件

# 2. 数据加载和预处理
titles = data["item_name"].tolist()  # 加载标题数据
labels_brand = data["brand"].tolist()  # 加载品牌标签数据
labels_model = data["model"].tolist()  # 加载型号标签数据
labels_category = data["category"].tolist()  # 加载种类标签数据


# 进行标题预处理，例如去除特殊字符、标点符号等

# 3. 特征提取
vectorizer = CountVectorizer()  # 使用词袋模型进行特征提取
X = vectorizer.fit_transform(titles)  # 转换标题文本为向量表示


label_encoder_brand = LabelEncoder()  # 创建品牌标签编码器
y_brand = label_encoder_brand.fit_transform(labels_brand)  # 对品牌标签进行编码

# 对型号和种类标签进行类似的编码操作
label_encoder_model = LabelEncoder()
y_model = label_encoder_model.fit_transform(labels_model)


label_encoder_category = LabelEncoder()
y_category = label_encoder_category.fit_transform(labels_category)

# 4. 模型训练
X_train, X_test, y_train_brand, y_test_brand, y_train_model, y_test_model, y_train_category, y_test_category = train_test_split(
    X, y_brand, y_model, y_category, test_size=0.2, random_state=42)

model_brand = MultinomialNB()  # 创建品牌分类模型
model_brand.fit(X_train, y_train_brand)  # 训练品牌分类模型

model_model = MultinomialNB()  # 创建型号分类模型
model_model.fit(X_train, y_train_model)  # 训练型号分类模型

model_category = MultinomialNB()  # 创建种类分类模型
model_category.fit(X_train, y_train_category)  # 训练种类分类模型



new_title = "小雨越南尾货已过检过验新版配饰黑银小号原厂小羊皮双大搭配三色链条的设计彰显着的不平凡大大的菱格蓬松慵懒的包型"  # 输入新的商品名
new_title_vector = vectorizer.transform([new_title])  # 转换新的商品名为特征向量

predicted_brand = label_encoder_brand.inverse_transform(model_brand.predict(new_title_vector))
predicted_model = label_encoder_model.inverse_transform(model_model.predict(new_title_vector))
predicted_category = label_encoder_category.inverse_transform(model_category.predict(new_title_vector))

print("商品标题：",new_title)
print("分类结果：")
print("品牌:", predicted_brand)
print("型号:", predicted_model)
print("种类:", predicted_category)
