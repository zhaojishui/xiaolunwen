__all__ = ['ATIO']

from StudentModel.student import studentmodel
from TeacherModel import *
from StudentModel  import *


class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'STUDENT': studentmodel,

        }

    # 修改为正确的映射
    def getTrain(self, args):
        if args.model_name == 'STUDENT':
            from teacherrun import studentmodel  # 引入专门的训练器类
            return studentmodel(args)