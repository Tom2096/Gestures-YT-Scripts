#!coding=utf-8
import json

from IPython import embed
import numpy as np

class Dataset(object):

    labels_dict = {
        'fist': 0,
        'open': 1,
        'left': 2,
        'right': 3,
        'ok': 4,
    }

    def __init__(self, conf):
        with open(conf['src']) as f:
            self.data = json.load(f)['dataset']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        points = np.array(self.data[idx]['data'])
        label = self.labels_dict[self.data[idx]['label']]
        
        return (self.norm(points), label)

    def norm(self, points):
        x_max, y_max = np.amax(points, axis=0)
        x_min, y_min = np.amin(points, axis=0)
        w = x_max - x_min
        h = y_max - y_min      

        for i in range(len(points)):
            points[i][0] = (points[i][0] - x_min) / w
            points[i][1] = (points[i][1] - y_min) / h

        return points
