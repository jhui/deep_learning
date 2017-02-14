import numpy as np

from enum import Enum

class SIDE(Enum):
    DEALER = 0
    PLAYER = 1

class ACTION(Enum):
    HIT = 0
    STICK = 1

class STATUS(Enum):
    CONTINUE = 1
    TERMINAL = 0

class State:
    def __init__(self):
        self.dealer_cards = [np.random.randint(1,11)]
        self.player_cards = [np.random.randint(1,11)]

    def player_sum(self):
        return sum(self.player_cards)

    def dealer_sum(self):
        return sum(self.dealer_cards)

    def __str__(self):
        return str((self.player_cards, self.player_sum(), self.dealer_cards, self.dealer_sum()))

    def draw(self, player=SIDE.PLAYER):
        new_card = np.random.randint(1, 11)
        if np.random.randint(1, 4)==3:
            new_card = - new_card
        if player==SIDE.PLAYER:
            self.player_cards.append(new_card)
        else:
            self.dealer_cards.append(new_card)

    def step(self, action=ACTION.HIT):
        if action == ACTION.HIT:
            self.draw()
            if 0 < self.player_sum() <= 21:
                return STATUS.CONTINUE, 0
            else:
                return STATUS.TERMINAL, -1.0
        else:
            while(0 < self.dealer_sum() < 17):
                self.draw(SIDE.DEALER)
                if self.dealer_sum() < 1 or self.dealer_sum() > 21:
                    return STATUS.TERMINAL, 1.0
            return STATUS.TERMINAL, np.sign(self.player_sum() - self.dealer_sum())

if __name__ == "__main__":
    game = State()
    status = STATUS.CONTINUE
    print(game)
    while status==STATUS.CONTINUE:
        action = ACTION.STICK if game.player_sum() > 14 else ACTION.HIT
        status, result = game.step(action)
        print(game)
        print(result)
