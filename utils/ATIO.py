__all__ = ['ATIO']

from StudentModel.student import studentmodel
from TeacherModel import *
from StudentModel  import *


class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'STUDENT': studentmodel,

        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)