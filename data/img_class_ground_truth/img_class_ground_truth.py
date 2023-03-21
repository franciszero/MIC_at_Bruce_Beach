import fiftyone as fo
import numpy as np
import seaborn as sns
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
from pandas import IndexSlice

from models.utils.Consts import MODEL_LIST

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)
pd.set_option('display.precision', 3)


def generate_img_clf_gt(metrics_dic, model_name):
    try:
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
    except Exception:
        raise


def new_metric_frame():
    return pd.DataFrame(data=np.zeros(len(MODEL_LIST) * 6).reshape(-1, len(MODEL_LIST)),
                        columns=MODEL_LIST,
                        index=['accuracy', 'precision', 'recall', 'f1score', 'support', 'mAP'],
                        ).copy()


dict_metrics = defaultdict(lambda: new_metric_frame())
for model_id in range(len(MODEL_LIST)):
    generate_img_clf_gt(dict_metrics, MODEL_LIST[model_id])

path = './'
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
tmp.reset_index('metric').pivot(columns='metric', values='best_model_name') \
    .to_csv(path + 'labels.csv', float_format='%.3f')

# visualization
plot_df = pd.read_csv('./data/img_class_ground_truth/accuracy.csv')
plot_df = plot_df.sort_values(by=list(MODEL_LIST)[::-1], ascending=False)
plot_df = plot_df.set_index(['file'])
plot_df = plot_df.stack()
plot_df.index = plot_df.index.rename('model', level=1)
plot_df.name = 'accuracy'
plot_df = plot_df.reset_index()
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
sns.lineplot(data=plot_df, x="file", y='accuracy', hue='model', style="model",
             markers=False, dashes=False, lw=1, ax=ax)
ax.set_title('Model accuracy comparison')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()

plt.savefig('./data/img_class_ground_truth/img_class_ground_truth.png')
