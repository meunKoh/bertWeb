import os
from flask import Flask, render_template, request, render_template_string
from werkzeug.utils import secure_filename
import visualization
import tensorflow as tf
import torch
import numpy as np

app = Flask(__name__, static_folder="static")
UPLOAD_DIR = 'C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
model_map = {
    'bert' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\result\\LM310K-MC25K"
    , 'bert10k' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\test\\mc-out\\kt-10k-20"
    , 'bert20k' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\test\\mc-out\\kt-20k-20"
    , 'bert30k' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\test\\mc-out\\kt-30k-20"
    , 'ft10k' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\f_models\\f_01.model"
    , 'ft20k' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\f_models\\f_02.model"
    , 'ft30k' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\f_models\\f_03.model"
    , 'cnn10k' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\f_models\\c_01.h5"
    , 'cnn20k' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\f_models\\c_02.h5"
    , 'cnn30k' : "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\f_models\\c_03.h5"
}

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'GET':
        return render_template("main.html")
    elif request.method == 'POST':

        model = request.form["models"]
        model_path = model_map[model]

        if model in ['bert','bert10k','bert20k','bert30k']:
            import classifier
            bert_classifier = classifier.Predictor(model_path)
            return do_classification(bert_classifier, app, request)
        # elif model in ['cnn10k','cnn30k']:
        #     import classifier_cnn
        #     cnn_classifier = classifier_cnn.Predictor(model_path)
        #     return do_classification(cnn_classifier, app, request)
        elif model in ['ft10k', 'ft20k', 'ft30k']:
            import fasttext
            model = fasttext.load_model(model_path)
            return do_fasttext(model, k=5, app=app, request=request)
        else:
            return render_template("main.html")


@app.route('/lime', methods=['POST'])
def lime_result():
    model = request.form["model"]
    model_path = model_map[model]

    text = request.form["text"]
    actual_label = request.form["actual_label"]
    pred_label = request.form["pred_label"]

    LABEL_LIST_KOREAN = ['경제민주화', '교통/건축/국토', '농산어촌', '문화/예술/체육/언론', '미래', '반려동물', '보건복지', '성장동력'
        , '안전/환경', '외교/통일/국방', '육아/교육', '인권/성평등', '일자리', '저출산/고령화대책', '정치개혁', '행정']
    actual_label_num = LABEL_LIST_KOREAN.index(actual_label)
    pred_label_num = LABEL_LIST_KOREAN.index(pred_label)


    if model in ['bert', 'bert10k', 'bert20k', 'bert30k']:
        import classifier
        bert_classifier = classifier.Predictor(model_path)
        html = visualization.get_lime_results(text
                                              , bert_classifier.predict
                                              , actual_label=actual_label_num
                                              , pred_label=pred_label_num)
        return render_template_string(html)

    elif model in ['ft10k', 'ft20k', 'ft30k']:
        import fasttext
        ft_classifier = fasttext.load_model(model_path)
        html = visualization.get_lime_results(text
                                              , lambda x: fasttext_prediction_in_sklearn_format(ft_classifier, x)
                                              , actual_label=actual_label_num
                                              , pred_label=pred_label_num)
        return render_template_string(html)
    else:
        return render_template("main.html")


def fasttext_prediction_in_sklearn_format(classifier, texts):
    res = []
    labels, probabilities = classifier.predict(texts, 16)
    labels = list(labels)
    labels = [label[9:] for label in labels]
    probabilities = list(probabilities)

    for label, probs, text in zip(labels, probabilities, texts):
        order = np.argsort(np.array(label))
        res.append(probs)

    return np.array(res)


def do_classification(classifier, app, request):

    uploaded_files = request.files.getlist("filesToUpload")

    filepaths = []
    filetitles = []
    texts = []
    answers = []
    all_labels = []
    all_probs = []
    corrects = []
    model = request.form["models"]

    for file in uploaded_files:
        # 업로드된 파일을 UPLOAD_FOLDER에 저장
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = UPLOAD_DIR + '\\' + filename
        filepaths.append(filepath)

        curr_filetitles, curr_texts, curr_answers\
            , curr_all_labels, curr_all_probs, curr_corrects = \
            visualization.classification_result_of_file(filepath, classifier)



        filetitles.extend(curr_filetitles)
        texts.extend(curr_texts)
        answers.extend(curr_answers)
        all_labels.extend(curr_all_labels)
        all_probs.extend(curr_all_probs)
        corrects.extend(curr_corrects)

    corrects_num = len([c for c in corrects if c == 'O'])
    torch.cuda.empty_cache()

    return render_template("main.html"
                           , len=len(filetitles)
                           , filenames=filetitles
                           , texts=texts
                           , answers=answers
                           , all_labels=all_labels
                           , all_probs=all_probs
                           , corrects=corrects
                           , corrects_num=corrects_num
                           , selected_model=model)

def do_fasttext(classifier, k, app, request):
    uploaded_files = request.files.getlist("filesToUpload")

    filepaths = []
    filetitles = []
    texts = []
    answers = []
    all_labels = []
    all_probs = []
    corrects = []
    model = request.form["models"]

    for file in uploaded_files:
        # 업로드된 파일을 UPLOAD_FOLDER에 저장
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = UPLOAD_DIR + '\\' + filename
        filepaths.append(filepath)

        curr_filetitles, curr_texts, curr_answers \
            , curr_all_labels, curr_all_probs, curr_corrects = \
            visualization.fasttext_result_of_file(filepath, classifier)

        filetitles.extend(curr_filetitles)
        texts.extend(curr_texts)
        answers.extend(curr_answers)
        all_labels.extend(curr_all_labels)
        all_probs.extend(curr_all_probs)
        corrects.extend(curr_corrects)

    corrects_num = len([c for c in corrects if c == 'O'])
    torch.cuda.empty_cache()

    return render_template("main.html"
                           , len=len(filetitles)
                           , filenames=filetitles
                           , texts=texts
                           , answers=answers
                           , all_labels=all_labels
                           , all_probs=all_probs
                           , corrects=corrects
                           , corrects_num=corrects_num
                           , selected_model=model)


if __name__ == "__main__":
    app.run(port=8090, debug=True)
