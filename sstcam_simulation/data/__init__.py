from os.path import join, dirname


def get_data(path):
    return join(dirname(__file__), path)
