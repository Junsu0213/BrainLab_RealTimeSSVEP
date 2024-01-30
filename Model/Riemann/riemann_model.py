# -*- coding:utf-8 -*-
from pyriemann.estimation import *
from pyriemann.classification import *
from sklearn.linear_model import LogisticRegression


class XdawnRG(object):
    def __init__(self, nfilter):
        self.xdawn = XdawnCovariances(nfilter, estimator='lwf')
        self.clf = LogisticRegression(max_iter=250, multi_class='multinomial')
        self.classifier = TSclassifier(metric='riemann', clf=self.clf)

    def train(self, x, y):
        # [Stage 1] => xDAWN Spatial Filtering
        x_ = self.xdawn.fit_transform(x, y)

        # [stage 2] => classification
        self.classifier.fit(x_, y)
        return self

    def predicate(self, x):
        # [Stage 1] => xDAWN Spatial Filtering
        x_ = self.xdawn.transform(x)

        # [stage 2] => classification
        out, prob = self.classifier.predict(x_), self.classifier.predict_proba(x_)
        return out, prob
