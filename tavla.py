import argparse
from game import environment



def _main_(args):
    board = environment()
    board.start()



ARGPARSER = argparse.ArgumentParser()
if __name__ == '__main__':
    _main_(ARGPARSER.parse_args())
