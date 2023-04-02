import fiftyone as fo
import numpy as np
import seaborn as sns
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
from pandas import IndexSlice
from fiftyone import ViewField as F
import os
import plotly
import kaleido
import plotly.io as pio
import sys
sys.path.append("../..")

from models.utils.Consts import MODEL_LIST

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)
pd.set_option('display.precision', 3)


def visualize_mAP_with_plotly(default_dataset, model_name):
    # predictions_view = default_dataset.take(default_dataset.count(), seed=51)
    # high_conf_view = predictions_view.filter_labels("predictions", F("confidence") > 0.5, only_matches=False)
    results = default_dataset.view().evaluate_detections(
        "predictions",
        gt_field="detections",
        eval_key="eval",
        compute_mAP=True,
    )
    # Get the 10 most common classes in the dataset
    counts = default_dataset.count_values("detections.detections.label")
    classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

    # Print a classification report for the top-10 classes
    results.print_report(classes=classes_top10)
    print("mAP score: " % results.mAP())
    plot = results.plot_pr_curves(classes=["1"])
    plot.update_layout(
        title_text=model_name,
        font_family="Courier New",
        font_color="blue",
        title_font_family="Times New Roman",
        title_font_color="red",
        legend_title_font_color="green"
    )
    plot.show()
    # # running kaleido in a venv meets this bug: https://github.com/plotly/Kaleido/issues/78
    # filepath = os.getcwd() + '/models/' + model_name + '/' + model_name + '.png'
    # pio.write_image(plot, filepath, format='png', engine='kaleido')


def generate_img_clf_gt(metrics_dic, model_name):
    try:
        dataset = fo.load_dataset(model_name)
        # # # running kaleido in a venv meets this bug: https://github.com/plotly/Kaleido/issues/78
        # visualize_mAP_with_plotly(dataset, model_name)
        for i, sample in enumerate(dataset):
            print("[%d/%d] %s" % (i, dataset.count(), model_name))
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
    except Exception:  # model_name is not exist
        raise


def new_metric_frame():
    return pd.DataFrame(data=np.zeros(len(MODEL_LIST) * 6).reshape(-1, len(MODEL_LIST)),
                        columns=MODEL_LIST,
                        index=['accuracy', 'precision', 'recall', 'f1score', 'support', 'mAP'],
                        ).copy()


# visualization
def metric_visualization(filename, dpi=150):
    name_of_file = filename.split('.', 1)[0]
    plot_df = pd.read_csv(filename)
    x = int(plot_df.index.size / 6)
    y = int(x / 4)
    plot_df = plot_df.sort_values(by=list(MODEL_LIST)[::-1], ascending=False)
    plot_df = plot_df.set_index(['file'])
    plot_df = plot_df.stack()
    plot_df.index = plot_df.index.rename('model', level=1)
    plot_df.name = name_of_file
    plot_df = plot_df.reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(x, y), dpi=dpi)
    sns.lineplot(data=plot_df[plot_df['model'] != MODEL_LIST[-1]], x="file", y=name_of_file, hue='model', style="model",
                 markers=False, dashes=False, lw=1, palette=sns.color_palette("bright", 8), ax=ax)
    sns.lineplot(data=plot_df[plot_df['model'] == MODEL_LIST[-1]], x="file", y=name_of_file, hue='model', style="model",
                 markers='*', dashes=False, lw=1.5, palette=['r'], ax=ax)
    ax.set_title('Model ' + name_of_file + ' comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=8)
    plt.tight_layout()
    plt.savefig('./' + name_of_file + '.jpg')


path = './'
dict_metrics = defaultdict(lambda: new_metric_frame())
for model_id in range(len(MODEL_LIST)):
    generate_img_clf_gt(dict_metrics, MODEL_LIST[model_id])

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

metric_visualization('accuracy.csv', dpi=150)
metric_visualization('mAP.csv', dpi=150)
