from game import environment
import wandb

def train():
    board = environment()
    board.start()


wandb.agent('hakanonal/tavla2/22zy2fld',function=train)    
