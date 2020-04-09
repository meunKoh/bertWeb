#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import InputExample


class Predictor(object):

    def __init__(self, model_path):
        task_name = "user-mc"
        #model_path = "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\result\\LM310K-MC25K"
        sp_model_path = "C:\\Users\\kkk\\PycharmProjects\\hug-face\\hug\\result\\sp-all-30000.model"
        self.device = torch.device("cuda")  # "cpu"
        processor = processors[task_name]()
        self.label_list = processor.get_labels()
        self.output_mode = output_modes[task_name]
        self.max_seq_length = 512
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False, sp_model_path=sp_model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)


    def predict(self, texts):
        # start_time = time.time()
        self.model.eval()
        examples = []
        for text in texts:
            example = InputExample(None, text, label="0")
            examples.append(example)
        features = convert_examples_to_features(examples,
                                                self.tokenizer,
                                                label_list=self.label_list,
                                                max_length=self.max_seq_length,
                                                output_mode=self.output_mode,
                                                pad_token=
                                                self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                )
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # elapsed_time = time.time() - start_time
        logits = outputs[0]
        prob = F.softmax(logits, dim=1)
        return prob.cpu().numpy()




