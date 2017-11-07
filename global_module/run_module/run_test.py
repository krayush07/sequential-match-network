from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from global_module.settings_module import Dictionary
from global_module.implementation_module import Test


#########################################################
# Utility function to load training vocab files
#########################################################

# def load_dictionary():
#     """
#     Utility function to load training vocab files
#     :return:
#     """
#     return set_dict.Dictionary('TE')


def initialize_test_session():
    dict_obj, test_obj = test_util()
    session, mtest = test_obj.init_test(dict_obj)
    return session, mtest, dict_obj, test_obj


def call_test(session, mtest, dict_obj, test_obj):
    test_obj.run_test(session, mtest, dict_obj)


def test_util():
    """
    Utility function to execute the testing pipeline
    :return:
    """
    dict_obj = Dictionary('TE')
    test_obj = Test()
    return dict_obj, test_obj


def main():
    """
    Starting module for testing
    :return:
    """
    print('STARTING TESTING')

    session, mtest, dict_obj, test_obj = initialize_test_session()
    call_test(session, mtest, dict_obj, test_obj)


if __name__ == '__main__':
    main()
