from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from global_module.implementation_module import Train
from global_module.pre_processing_module import BuildWordVocab, GenerateLabel, SampleTrainingData
from global_module.settings_module import Dictionary


# def load_dictionary():
#     """
#     Utility function to load training vocab files
#     :return:
#     """
#     return set_dict.Dictionary()


def call_train(dict_obj):
    """
    Utility function to execute main training module
    :param dict_obj: dictionary object
    :return: None
    """
    Train().run_train(dict_obj)
    return


def train_util():
    """
    Utility function to execute the training pipeline
    :return: None
    """
    SampleTrainingData().util()
    BuildWordVocab().util()
    GenerateLabel().util()
    dict_obj = Dictionary()
    call_train(dict_obj)
    return None


def main():
    """
    Starting module for CLSTM testing
    :return:
    """
    print('STARTING TRAINING')
    train_util()
    return None


if __name__ == '__main__':
    main()
