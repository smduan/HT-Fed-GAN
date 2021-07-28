import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, r2_score, recall_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from constants import CATEGORICAL, COL_TYPE_ALL, CONTINUOUS, DATASET_NAME, ORDINAL

LOGGER = logging.getLogger(__name__)
DATA_PATH = './{0}'.format(DATASET_NAME)  # json文件放置的目录


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def load_dataset(name, benchmark=False):
    # LOGGER.info('Loading dataset %s', name)
    local_path = os.path.join(DATA_PATH, name)
    meta = _load_json(local_path)
    categorical_columns, ordinal_columns = _get_columns(meta)

    test_data = pd.read_csv('./{0}/{0}_test.csv'.format(DATASET_NAME), dtype='str', header=0, error_bad_lines=False, lineterminator="\n", encoding='utf-8')  # /修改/训练数据路径
    test_data = test_data.apply(lambda x: x.str.strip(' \t.'))
    test = read_data(test_data)

    if benchmark:
        return test, meta, categorical_columns, ordinal_columns

    return categorical_columns, ordinal_columns


#################################################
def project_table(data, meta):
    values = np.zeros(shape=data.shape, dtype='float32')

    for id_, info in enumerate(meta):
        if info['type'] == CONTINUOUS:
            values[:, id_] = data.iloc[:, id_].values.astype('float32')
        else:
            mapper = dict([(item, id) for id, item in enumerate(info['i2s'])])
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
            values[:, id_] = mapped
    return values


def read_data(df):
    col_type = COL_TYPE_ALL[DATASET_NAME]

    meta = []
    for id_, info in enumerate(col_type):
        if info[1] == CONTINUOUS:
            meta.append({
                "name": info[0],
                "type": info[1],
                "min": np.min(df.iloc[:, id_].values.astype('float')),
                "max": np.max(df.iloc[:, id_].values.astype('float'))
            })
        else:
            if info[1] == CATEGORICAL:
                value_count = list(dict(df.iloc[:, id_].value_counts()).items())  # 同一列不同取值及其数量
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))  # 取值提出来
            else:
                mapper = info[2]

            meta.append({
                "name": info[0],
                "type": info[1],
                "size": len(mapper),
                "i2s": mapper
            })
    # print(meta)
    tdata = project_table(df, meta)
    return tdata


#######################################################################
def Evaluate_binary_classification(train, test, metadata, label_column):
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata, label_column)

    performance = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        LOGGER.info('Evaluating using binary classifier %s', model_repr)
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='binary')
        recall = recall_score(y_test, pred)
        precision = precision_score(y_test, pred)

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                "f1": f1,
                "recall": recall,
                "precision": precision
            }
        )

    return pd.DataFrame(performance)


#########################################################################
# regression
_MODELS = {
    'binary_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 15,
                'class_weight': 'balanced',
            }
        },
        {
            'class': AdaBoostClassifier,
        },
        {
            'class': LogisticRegression,
            'kwargs': {
                'solver': 'lbfgs',
                'n_jobs': 2,
                'class_weight': 'balanced',
                'max_iter': 50
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (50,),
                'max_iter': 50
            },
        }
    ],
    'multiclass_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 30,
                'class_weight': 'balanced',
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (100,),
                'max_iter': 50
            },
        },
        {
            'class': RandomForestClassifier
        },
        {
            'class': GradientBoostingClassifier
        }
    ],
    'regression': [
        {
            'class': Lasso,
        },
        {
            'class': MLPRegressor,
            'kwargs': {
                'hidden_layer_sizes': (100,),
                'max_iter': 50
            },
        },
        # {
        #     'class':SVR,  #径向基核函数
        #     'kwargs':{
        #         'kernel' : 'rbf'
        #     }
        # },
        {
            'class': GradientBoostingRegressor,  # 提升树
        },
        {
            'class': RandomForestRegressor
        }
    ]
}


class FeatureMaker:

    def __init__(self, metadata, label_column, label_type='int', sample=50000):
        self.columns = metadata['columns']
        # print(self.columns)
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):
        data = data.copy()
        # print(data)
        # print('$$$$$$$$$$$$$$$$$$$$$$$')
        np.random.shuffle(data)
        data = data[:self.sample]  # 取前50000条？
        # print(data)

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            # print(col)
            # print(index)
            # print(cinfo)
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    # print(cinfo['name'])
                    # print(self.label_type)
                    # print(col)
                    labels = col.astype(int)
                    # print(labels)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) * 5

            elif cinfo['type'] == ORDINAL:
                feature = col

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, test, metadata, label_column):
    fm = FeatureMaker(metadata, label_column)
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)

    return x_train, y_train, x_test, y_test, _MODELS[metadata['problem_type']]


def _evaluate_regression(train, test, metadata, label_column):
    x_train, y_train, x_test, y_test, regressors = _prepare_ml_problem(train, test, metadata, label_column)
    regressors = _MODELS['regression']

    performance = []
    y_train = np.log(np.clip(y_train, 1, 20000))
    y_test = np.log(np.clip(y_test, 1, 20000))
    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        LOGGER.info('Evaluating using regressor %s', model_repr)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        # print(y_test)
        # print(pred)
        performance.append(
            {
                "name": model_repr,
                "r2": r2,
                "mae": mae
            }
        )

    return pd.DataFrame(performance)


#########################################################################
# 多分类
def _evaluate_multi_classification(train, test, metadata, label_column):
    """Score classifiers using f1 score and the given train and test data.

    Args:
        x_train(numpy.ndarray):
        y_train(numpy.ndarray):
        x_test(numpy.ndarray):
        y_test(numpy):
        classifiers(list):

    Returns:
        pandas.DataFrame
    """
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata, label_column)
    classifiers = _MODELS['multiclass_classification']

    performance = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        LOGGER.info('Evaluating using multiclass classifier %s', model_repr)
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average='macro')
        micro_f1 = f1_score(y_test, pred, average='micro')

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            }
        )

    return pd.DataFrame(performance)


##########################################################################


if __name__ == '__main__':
    test, meta, categoricals, ordinals = load_dataset('{0}.json'.format(DATASET_NAME), benchmark=True)  # 读入训练数据以及adult数据集元数据
    # 从csv文件中读取生成数据
    synthesized_data = pd.read_csv("./{0}_fake.csv".format(DATASET_NAME), dtype='str', header=0, error_bad_lines=False, lineterminator="\n")  # /修改/生成数据路径
    synthesized_data = synthesized_data.apply(lambda x: x.str.strip(' \t.'))
    synthesized = read_data(synthesized_data)
    # 以下代码留取想训练的属性
    if DATASET_NAME == 'adult':
        scores_income_bracket = Evaluate_binary_classification(synthesized, test, meta, 'label')  # 注意：属性名称应与meta中的名称保持一致
        print(scores_income_bracket)
        scores_hours_per_week = _evaluate_regression(synthesized, test, meta, 'hours-per-week')
        print(scores_hours_per_week)
    elif DATASET_NAME == 'clinical':
        scores_DEATH_EVENT = Evaluate_binary_classification(synthesized, test, meta, 'DEATH_EVENT')
        print(scores_DEATH_EVENT)
        scores_age = _evaluate_regression(synthesized, test, meta, 'age')
        print(scores_age)
    elif DATASET_NAME == 'covtype':
        scores_label = _evaluate_multi_classification(synthesized, test, meta, 'label')
        print(scores_label)
        scores_Elevation = _evaluate_regression(synthesized, test, meta, 'Elevation')
        print(scores_Elevation)
    elif DATASET_NAME == 'credit':
        scores_label = Evaluate_binary_classification(synthesized, test, meta, 'label')
        print(scores_label)
        scores_Amount = _evaluate_regression(synthesized, test, meta, 'Amount')
        print(scores_Amount)
    elif DATASET_NAME == 'intrusion':
        scores_label = _evaluate_multi_classification(synthesized, test, meta, 'label')
        print(scores_label)
        scores_count = _evaluate_regression(synthesized, test, meta, 'count')
        print(scores_count)
