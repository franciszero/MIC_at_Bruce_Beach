import fiftyone as fo
import numpy as np
import seaborn as sns
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from pandas import IndexSlice
from fiftyone import ViewField as F
import os
import plotly
import kaleido
import plotly.io as pio

from models.utils.Consts import MODEL_LIST
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter, FixedLocator

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)
pd.set_option('display.precision', 3)

test = pd.DataFrame(columns=["Image", "ground truth", "train18", "train31"], dtype=int, )
ds16 = fo.load_dataset('YOLOv8x_BB158_500_t16_b')
ds31 = fo.load_dataset('YOLOv8x_BB426_760_t31_b')
for i, (s16, s31) in enumerate(zip(ds16, ds31)):
    if s16.filename == s31.filename:
        idx = test.index.values.size
        test.loc[idx, "Image"] = s31.filename
        if s16.detections is None and s31.detections is None:
            test.loc[idx, "ground truth"] = 0
        else:
            gt16 = s16.detections.detections
            gt31 = s31.detections.detections
            if len(gt16) == len(gt31):
                test.loc[idx, "ground truth"] = len(gt16)
            else:
                print(s16.filename, len(gt16), " != ", len(gt31))

        pred16 = s16.predictions.detections
        if pred16 is None:
            test.loc[idx, "train18"] = 0
        else:
            test.loc[idx, "train18"] = len(pred16)

        pred31 = s31.predictions.detections
        if pred31 is None:
            test.loc[idx, "train31"] = 0
        else:
            test.loc[idx, "train31"] = len(pred31)

        print(s16.filename, s31.filename, )
    else:
        print(s16.filename,
              len(s16.detections.detections),
              len(s16.predictions.detections),
              s31.filename,
              len(s31.detections.detections),
              len(s31.predictions.detections), )
        break

test_sort = test.sort_values(by=['ground truth', 'train31', 'train18'], ascending=False)
tmp = test_sort.set_index(['Image']).stack()
tmp = tmp.rename_axis(index=['Image', 'class'])
tmp.name = 'People counting'
tmp = tmp.reset_index()

fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=200)
sns.lineplot(tmp, x=tmp['Image'], y=tmp['People counting'], hue=tmp['class'],
             markers=False, dashes=False, lw=1, palette=sns.color_palette("bright", 8), ax=ax)
ax.set_title('Model predictions comparison')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=8)
ax.grid()
plt.tight_layout()
# plt.show()
plt.savefig('./data/img_class_ground_truth/Model predictions comparison.jpg')

print('')
