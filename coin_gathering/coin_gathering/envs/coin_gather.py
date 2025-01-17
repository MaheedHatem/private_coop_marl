import gym
from gym import spaces
from PIL import Image, ImageColor
from matplotlib import pyplot as plt
import numpy as np
from gym.utils import seeding

EMPTY = 0
AGENT = 1
COIN = 2
STAR = 3
BOMB = 4
OBJECT_COUNT = BOMB + 1
COIN_REWARD = 1
BOMB_REWARD = -0.5
RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3
ACTIONS_COUNT = DOWN + 1
ADD_COIN = 0
ADD_BOMB = 1
ADD_STAR = 2
DO_NOTHING = 3
LEVER_EFFECTS = DO_NOTHING + 1

class CoinGatherEnv(gym.Env):
    def __init__(self, grid_width=7, grid_height=9, N=3, max_episode_steps=1000,
                 levers_prob_self=[[0.8, 0.2, 0, 0], [0.6, 0.3, 0.1, 0]],
                 levers_prob_other=[[0.2, 0.7, 0.1, 0], [0.5, 0.4, 0.1, 0]], render_mode=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.N = N
        self.max_episode_steps = max_episode_steps
        self.levers_prob_self = np.array(levers_prob_self)
        self.levers_prob_other = np.array(levers_prob_other)

        self.action_count = ACTIONS_COUNT + len(self.levers_prob_self)
        self.observation_space = spaces.Tuple(tuple([spaces.MultiDiscrete([self.grid_width, self.grid_height,
                                                       2, 2 * self.grid_width - 1,
                                                       2 * self.grid_height - 1, 2,
                                                       2 * self.grid_width - 1, 2 * self.grid_height - 1] +
                                                      [2] * self.grid_width)] * self.N))
        self.action_space = spaces.Tuple(tuple([spaces.Discrete(self.action_count)] * self.N))
        self.levers_id =[i + ACTIONS_COUNT for i in range(len(self.levers_prob_self))]
        assert len(self.levers_prob_self) == len(self.levers_prob_other), 'Lever probabilities have different lengths'
        assert self.levers_prob_self.shape[1] == LEVER_EFFECTS, f'Each lever should have {LEVER_EFFECTS:.d} probabilities'
        assert self.levers_prob_self.shape[1] == self.levers_prob_other.shape[1], 'Single lever probabilities do not match in length'
        assert len(self.levers_prob_self) >= 1, 'At least one lever should be provided'
        assert len(self.levers_prob_self) <= self.grid_height // 2, 'Number of levers should not be greater than half the grid\'s height'
        assert np.all(np.abs(np.sum(levers_prob_self, axis=1)-1)<1e-10), 'Lever self probabilities do not sum to one'
        assert np.all(np.abs(np.sum(levers_prob_other, axis=1)-1)<1e-10), 'Lever other probabilities do not sum to one'

        self.seed()

        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.time_step = 0
        self.agents = [i for i in range(self.N)]
        self.rooms = np.zeros((self.N, self.grid_height, self.grid_width), dtype=np.float32)
        self.rooms[:, 0, 0] = 1
        self.agents_positions_y = np.zeros(len(self.agents), dtype=np.int64)
        self.agents_positions_x = np.zeros(len(self.agents), dtype=np.int64)
        self.coin_avail = np.zeros(len(self.agents), dtype=np.int64)
        self.coin_pos_x = np.zeros(len(self.agents), dtype=np.int64)
        self.coin_pos_y = np.zeros(len(self.agents), dtype=np.int64)
        self.star_avail = np.zeros(len(self.agents), dtype=np.int64)
        self.star_pos_x = np.zeros(len(self.agents), dtype=np.int64)
        self.star_pos_y = np.zeros(len(self.agents), dtype=np.int64)
        self.bombs = np.zeros((len(self.agents), self.grid_width), dtype=np.int64)

        return self.get_obs()

    def step(self, actions):
        actions = np.squeeze(actions)
        assert len(actions) == len(self.agents)
        self.time_step += 1
        rewards = [0 for name in self.agents]
        terminations = [self.time_step == self.max_episode_steps for name in self.agents]
        pulled_lever = [None for name in self.agents]

        # first move agents
        for a in self.agents:
            action = actions[a]
            assert action < self.action_count, f"invalid action {action}"
            y, x = (self.agents_positions_y[a], self.agents_positions_x[a])
            newx = x
            newy = y
            if action == LEFT and x > 0 and self.rooms[a, y, x-1] not in self.levers_id:
                newx = x-1
            elif action == RIGHT and x < self.grid_width-1 and self.rooms[a, y, x+1] not in self.levers_id:
                newx = x+1
            elif action == DOWN and y < self.grid_height-1 and self.rooms[a, y+1, x] not in self.levers_id:
                newy = y+1
            elif action == UP and y > 0 and self.rooms[a, y-1, x] not in self.levers_id:
                newy = y-1
            # elif action == USE and (x < self.grid_width-1 and self.rooms[a, y, x+1] in self.levers_id):
            #     pulled_lever[a] = int(self.rooms[a, y, x+1] - OBJECT_COUNT)
            # elif action ==USE and (x > 0 and self.rooms[a, y, x-1] in self.levers_id):
            #     pulled_lever[a] = int(self.rooms[a, y, x-1] - OBJECT_COUNT)
            elif action in self.levers_id:
                pulled_lever[a] = action - ACTIONS_COUNT

            if(self.rooms[a, newy, newx] == COIN):
                rewards[a] += COIN_REWARD
                self.coin_avail[a] = 0
                self.coin_pos_x[a] = 0
                self.coin_pos_y[a] = 0
            elif(self.rooms[a, newy, newx] == BOMB):
                rewards[a] += BOMB_REWARD
                self.bombs[a, newx] = 0
            elif(self.rooms[a, newy, newx] == STAR):
                self.rooms[a, self.rooms[a] == BOMB] = EMPTY
                self.bombs[a,:] = 0
                self.star_avail[a] = 0
                self.star_pos_x[a] = 0
                self.star_pos_y[a] = 0
            # if(self.coin_avail[a]):
            #     rewards[a] += 1/(np.abs(newx-self.coin_pos_x[a]) + np.abs(newy-self.coin_pos_y[a]))
            self.rooms[a, y, x] = EMPTY
            self.rooms[a, newy, newx] = AGENT
            self.agents_positions_y[a] = newy
            self.agents_positions_x[a] = newx
        
        # apply pulled levers
        for a in self.agents:
            pulled_lever_index = pulled_lever[a]
            if pulled_lever_index != None:
                agents_act = self.np_random.choice(LEVER_EFFECTS, p=self.levers_prob_self[pulled_lever_index])
                others_act = self.np_random.choice(LEVER_EFFECTS, p=self.levers_prob_other[pulled_lever_index])
                if agents_act == ADD_COIN:
                    self.add_coin([a])
                elif agents_act == ADD_BOMB:
                    self.add_bomb([a])
                elif agents_act == ADD_STAR:
                    self.add_star([a])

                if others_act == ADD_COIN:
                    self.add_coin([j for j in range(self.N) if j != a])
                elif others_act == ADD_BOMB:
                    self.add_bomb([j for j in range(self.N) if j != a])
                elif others_act == ADD_STAR:
                    self.add_star([j for j in range(self.N) if j != a])

        return self.get_obs(), rewards, terminations, {}

    def render(self, mode='human'):
        img = Image.new(('RGB'), (self.grid_width * len(self.rooms) + len(self.rooms) + 1, self.grid_height + 2))
        pixels = img.load()

        for a in range(len(self.rooms)):
            start_col = a * (self.grid_width + 1) + 1
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    i = y + 1
                    j = start_col + x
                    if self.rooms[a, y, x] == EMPTY:
                        pixels[j, i] = ImageColor.getrgb("white")
                    elif self.rooms[a, y, x] == AGENT:
                        pixels[j, i] = ImageColor.getrgb("blue")
                    elif self.rooms[a, y, x] == COIN:
                        pixels[j, i] = ImageColor.getrgb("yellow")
                    elif self.rooms[a, y, x] == BOMB:
                        pixels[j, i] = ImageColor.getrgb("grey")
                    elif self.rooms[a, y, x] == STAR:
                        pixels[j, i] = ImageColor.getrgb("orange")

        plt.imshow(img)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plt.show()

    def get_obs(self):
        return tuple([np.concatenate([np.array([
            self.agents_positions_x[a],
            self.agents_positions_y[a],
            self.coin_avail[a],
            self.coin_pos_x[a] - self.agents_positions_x[a] if self.coin_avail[a] else 0,
            self.coin_pos_y[a] - self.agents_positions_y[a] if self.coin_avail[a] else 0,
            self.star_avail[a],
            self.star_pos_x[a] - self.agents_positions_x[a] if self.star_avail[a] else 0,
            self.star_pos_y[a] - self.agents_positions_y[a] if self.star_avail[a] else 0]),
            self.bombs[a]]
            ) for a in range(self.N)])

    def add_coin(self, agents):
        for a in agents:
            if(self.coin_avail[a] == 1):
                return
            
            y = self.agents_positions_y[a]
            agent_half = 0 if y <= self.grid_height//2 else 1
            if agent_half == 0:
                y_start = self.grid_height//2 + 1
                y_end = self.grid_height
            else:
                y_start = 0
                y_end = self.grid_height//2
            grid = np.mgrid[y_start: y_end, 0:self.grid_width]
            mask = self.rooms[a,y_start: y_end,0:self.grid_width]==EMPTY
            valid_xindices = grid[1][mask]
            valid_yindices = grid[0][mask]
            if(len(valid_xindices) == 0):
                return
            chosen_cell = self.np_random.choice(len(valid_xindices))
            coinx = valid_xindices[chosen_cell]
            coiny = valid_yindices[chosen_cell]
            assert(self.rooms[a, coiny, coinx] == EMPTY)
            self.rooms[a, coiny, coinx] = COIN
            self.coin_avail[a] = 1
            self.coin_pos_x[a] = coinx
            self.coin_pos_y[a] = coiny

    def add_star(self, agents):
        for a in agents:
            y = self.agents_positions_y[a]
            agent_half = 0 if y <= self.grid_height//2 else 1
            if agent_half == 1:
                y_start = self.grid_height//2 + 1
                y_end = self.grid_height
            else:
                y_start = 0
                y_end = self.grid_height//2
            grid = np.mgrid[y_start:y_end , 0:self.grid_width]
            mask = self.rooms[a,y_start:y_end,0:self.grid_width]==EMPTY
            valid_xindices = grid[1][mask]
            valid_yindices = grid[0][mask]
            if(len(valid_xindices) == 0):
                return
            chosen_cell = self.np_random.choice(len(valid_xindices))
            starx = valid_xindices[chosen_cell]
            stary = valid_yindices[chosen_cell]
            assert(self.rooms[a, stary, starx] == EMPTY)
            self.rooms[a, self.rooms[a] == STAR] = EMPTY
            self.rooms[a, stary, starx] = STAR
            self.star_avail[a] = 1
            self.star_pos_x[a] = starx
            self.star_pos_y[a] = stary

    def add_bomb(self, agents):
        for a in agents:
            grid = np.mgrid[self.grid_height//2 : self.grid_height//2 + 1, 0:self.grid_width]
            mask = self.rooms[a,self.grid_height//2: self.grid_height//2 + 1,0:self.grid_width]==EMPTY
            valid_xindices = grid[1][mask]
            valid_yindices = grid[0][mask]
            if(len(valid_xindices) == 0):
                return
            chosen_cell = self.np_random.choice(len(valid_xindices))
            bombx = valid_xindices[chosen_cell]
            bomby = valid_yindices[chosen_cell]
            assert(self.rooms[a, bomby, bombx] == EMPTY)
            self.rooms[a, bomby, bombx] = BOMB
            self.bombs[a, bombx] = 1