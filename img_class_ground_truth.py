import fiftyone as fo
import numpy as np
from collections import defaultdict
import pandas as pd
from pandas import IndexSlice

from models.utils.Consts import MODEL_LIST

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)
pd.set_option('display.precision', 3)


def generate_img_clf_gt(metrics_dic, model_name):
    dataset = fo.load_dataset(model_name)
    for sample in dataset:
        df = metrics_dic[sample.filename]
        if sample.get_field('detections') is None:
            lst = sample.get_field('predictions').get_field('detections')
            ap = 1.0 if lst == [] else 0.0
            [acc, pre, rec, f1, sup] = [1.] * 5 if lst == [] else [0.] * 5
        else:
            results = dataset.select(sample.id) \
                .evaluate_detections("predictions", gt_field="detections", iou=0.4,
                                     eval_key="eval", compute_mAP=True, )
            ap = results.mAP()
            [acc, pre, rec, f1, sup] = results.metrics().values()
        df.loc[:, model_name] = [acc, pre, rec, f1, sup, ap]
    pass


def new_metric_frame():
    return pd.DataFrame(data=np.zeros(3 * 6).reshape(-1, 3),
                        columns=MODEL_LIST,
                        index=['accuracy', 'precision', 'recall', 'f1score', 'support', 'mAP'],
                        ).copy()


dict_metrics = defaultdict(lambda: new_metric_frame())
for model_id in range(len(MODEL_LIST)):
    generate_img_clf_gt(dict_metrics, MODEL_LIST[model_id])

path = './data/img_class_ground_truth/'
tmp = pd.DataFrame(index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=[u'file', u'metric']),
                   columns=MODEL_LIST, dtype=float, )
for (n, a) in dict_metrics.items():
    for metric in a.index:
        mt = a[a.index == metric]
        tmp.loc[(n, metric), MODEL_LIST] = mt.values.flatten().round(3)
        tmp.loc[(n, metric), 'best_model_name'] = mt.T.idxmax().values[0]
tmp.to_csv(path + 'rawdata.csv', float_format='%.3f')

for metric in tmp.index.get_level_values('metric').unique():
    tmp.loc[IndexSlice[:, metric], MODEL_LIST].droplevel('metric').to_csv(path + '%s.csv' % metric, float_format='%.3f')
tmp.reset_index('metric').pivot(columns='metric', values='best_model_name')\
    .to_csv(path + 'labels.csv', float_format='%.3f')
