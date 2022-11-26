from game import Game
from gestures import Gestures
from time import sleep

if __name__ == '__main__':
    gestures = Gestures()
    gestures.start()
    sleep(2)
    game = Game(gestures)
