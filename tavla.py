import argparse
from game import game



def _main_(args):
    board = game()
    board.start()



ARGPARSER = argparse.ArgumentParser()
if __name__ == '__main__':
    _main_(ARGPARSER.parse_args())
