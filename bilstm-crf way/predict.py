from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
from evaluating import Metrics
from evaluate import ensemble_evaluate
import sys

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = './ckpts/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记

hmm_pred = []
crf_pred = []
lstm_pred = []
lstmcrf_pred = []
train_word_lists, train_tag_lists, word2id, tag2id = \
    build_corpus("train")
dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

def hmm_predict():
    print("hmm模型的预测结果...")
    hmm_model = load_model(HMM_MODEL_PATH)
    hmm_pred = hmm_model.test(test_word_lists,
                              word2id,
                              tag2id)
    return hmm_pred

def crf_predict():
    print("crf模型预测结果...")
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(test_word_lists)
    return crf_pred

def bilstm_predict():
    print("bilstm模型预测结果...")
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    bilstm_model = load_model(BiLSTM_MODEL_PATH)
    bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                   bilstm_word2id, bilstm_tag2id)
    return lstm_pred

def bilstm_crf_predict():
    print("bilstm+crf模型预测结果...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    test_word_lists_bilstm, test_tag_lists_bilstm = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists_bilstm, test_tag_lists_bilstm,
                                                      crf_word2id, crf_tag2id)
    return lstmcrf_pred


def extract_entities(tokens, tags):
    entities = []
    current_entity = None

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {'text': token, 'label': tag[2:]}
        elif tag.startswith('M-'):
            if current_entity and current_entity['label'] == tag[2:]:
                current_entity['text'] += token
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = None
        elif tag.startswith('E-'):
            if current_entity and current_entity['label'] == tag[2:]:
                current_entity['text'] += token
                entities.append(current_entity)
                current_entity = None
        elif tag.startswith('S-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {'text': token, 'label': tag[2:]}
            entities.append(current_entity)
            current_entity = None
        else:
            current_entity = None

    return entities


def get_segmentation_and_attributes(tokens, entities):
    segmentation = ' '.join(tokens)
    attributes = {}  # 用于存储实体属性

    for entity in entities:
        if entity['label'] not in attributes:  # 如果实体的标签不在属性中，则创建新的属性列表
            attributes[entity['label']] = []
        attributes[entity['label']].append(entity['text'])  # 将实体文本添加到对应标签的属性列表中

    return segmentation, attributes

    # 示例预测结果

def process_sentences(sentences, predictions):
    results = []  # 用于存储处理结果

    for sentence, tags in zip(sentences, predictions):
        tokens = [char for char in sentence]        # 将句子拆分为分词结果

        entities = extract_entities(tokens, tags)  # 提取实体

        segmentation, attributes = get_segmentation_and_attributes(tokens, entities)  # 获取分词结果和属性

        result = {
            'sentence': sentence,
            'segmentation': segmentation,
            'attributes': attributes
        }

        results.append(result)

    return results




if __name__ == "__main__":

    # 测试数据
    sentences = test_word_lists
    #可将hmm_pred替换为crf_pred、lstm_pred、lstmcrf_pred 
    predictions = bilstm_crf_predict()
    print(predictions)
    # 处理句子
    results = process_sentences(sentences, predictions)
    with open('result/bilstm_crf_result.txt','w') as file :
        sys.stdout = file
        # 打印结果
        for result in results:
            print("句子:", result['sentence'])
            print("分词结果:", result['segmentation'])
            print("属性:", result['attributes'])
            print()
        sys.stdout = sys.__stdout__
    print("内容已保存到bilstm_crf_result.txt中")