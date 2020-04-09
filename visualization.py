import pandas as pd
import numpy as np
import os
from lime.lime_text import LimeTextExplainer

LABEL_LIST_KOREAN = ['경제민주화', '교통/건축/국토', '농산어촌', '문화/예술/체육/언론', '미래', '반려동물', '보건복지', '성장동력'
    , '안전/환경', '외교/통일/국방', '육아/교육', '인권/성평등', '일자리', '저출산/고령화대책', '정치개혁', '행정']


def get_contents(row):
    answer = LABEL_LIST_KOREAN[row.category_label]
    return row.contents, answer


def show_probability(classifier, text):
    prob = classifier.predict([text])

    # 예측정확도 순으로 3개 라벨리스트를 추출
    sorted_labels = np.argsort(prob)[0][-3:][::-1].tolist()
    sorted_labelnames = [LABEL_LIST_KOREAN[num] for num in sorted_labels]

    # 예측정확도를 2자리수에서 반올림하여 반환
    sorted_probs = np.sort(prob)[0][-3:][::-1]
    round_probs = [int(round(prob*100, 0)) for prob in sorted_probs]

    return sorted_labelnames, round_probs


def classification_result_of_file(filepath, classifier):
    df = pd.read_table(filepath, delimiter='\t', index_col=0, encoding='utf-8')
    filetitle = os.path.basename(filepath)

    filetitles = []
    texts = []
    answers = []
    all_labels = []
    all_probs = []
    corrects = []

    for i in range(len(df)):
        text, answer = get_contents(df.iloc[i])
        if type(text) != str:
            continue
        labels, probs = show_probability(classifier, text)

        texts.append(text)
        answers.append(answer)
        all_labels.append(labels)
        all_probs.append(probs)

        if labels[0] == answer:
            corrects.append('O')
        else:
            corrects.append('X')

        if i==0 and len(df)==1:
            filetitles.append(filetitle)
        else:
            curr_filetitle = filetitle + '_' + str(i)
            filetitles.append(curr_filetitle)

    return filetitles, texts, answers, all_labels, all_probs, corrects

def fasttext_result_of_file(filepath, classifier):
    df = pd.read_table(filepath, delimiter='\t', index_col=0, encoding='utf-8')
    filetitle = os.path.basename(filepath)

    filetitles = []
    texts = []
    answers = []
    all_labels = []
    all_probs = []
    corrects = []

    for i in range(len(df)):
        text, answer = get_contents(df.iloc[i])
        if type(text) != str:
            continue
        labels, probs = classifier.predict(text, k=5)
        labels = list(labels)
        labels = [label[9:] for label in labels]
        probs = list(probs)
        # 예측정확도를 2자리수에서 반올림하여 반환
        probs = [int(round(prob * 100, 0)) for prob in probs]

        texts.append(text)
        answers.append(answer)
        all_labels.append(labels)
        all_probs.append(probs)

        if labels[0] == answer:
            corrects.append('O')
        else:
            corrects.append('X')

        if i==0 and len(df)==1:
            filetitles.append(filetitle)
        else:
            curr_filetitle = filetitle + '_' + str(i)
            filetitles.append(curr_filetitle)

    return filetitles, texts, answers, all_labels, all_probs, corrects


def get_lime_results(text, predictor, actual_label, pred_label, num_features=6, num_samples=85):
    explainer = LimeTextExplainer(class_names=LABEL_LIST_KOREAN)
    exp = explainer.explain_instance(text
                                     , predictor
                                     , num_features=num_features
                                     , labels=[actual_label, pred_label]
                                     , num_samples=num_samples)
    html = exp.as_html()
    html = html.replace("* _.templateSettings.interpolate = /{{([\s\S]+?)}}/g;", "")
    return html
