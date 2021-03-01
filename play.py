from gym_wrapper import transform_env
import torch
import gym
from networks import networkV1, networkV2
import torch.nn.functional as F
import time
import tkinter as tk
from PIL import Image, ImageTk
from math import sqrt, pow


class Agent:
    def __init__(self, model):
        # init vars
        self.model = model
        self.actions_count = model.actions_count

        # device - define and cast
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def choose_action(self, observation):
        # add dimension (batch) to match (batch, layer, height, width), transfer to GPU
        observation = observation.unsqueeze(0).to(self.device).float()

        # we do not compute gradients when choosing actions, hence no_grad
        with torch.no_grad():
            outActor, _ = self.model(observation)

        # transform output of forward pass, so probabilities will all add up to 1
        probs = F.softmax(outActor, dim=-1)

        # transfer to CPU after calculation
        probs = probs.cpu()

        # sort probabilities
        action = probs.multinomial(num_samples=1)

        # return highest probability
        return action[0].item()  # TODO extract confidence


def play():
    path = 'models/test2/test2_0_a2c.pt'
    actions = 5
    agent = Agent(networkV1(actions))
    agent.load_model(path)

    env = gym.make('MsPacman-v0')
    env_ai = transform_env(env, 4)
    observation = env.reset()
    observation_ai = env_ai.reset()
    done = False

    while not done:
        env_ai = transform_env(env, 4)
        env.render()

        action_ai = agent.choose_action(torch.from_numpy(observation_ai))
        _, _, done, _ = env.step(action_ai)

        time.sleep(1 / 60)  # FPS

# TODO put everything in a class
# TODO add threads
# TODO split for user control and AI suggestions
# TODO embed environment into GUI
# TODO create and embed control matrix into GUI
# TODO embed graph with AI certainty into GUI
# TODO embed graph with player vs ai decision divergence ?
# TODO add "autopilot" option


# Window parameters
game_width = 700
game_height = 700

suggestion_window_dim = 300
suggestion_window_center = suggestion_window_dim / 2
suggestion_circle_r = suggestion_window_dim / 10
suggestion_circle = [(suggestion_window_center - suggestion_circle_r),
                     (suggestion_window_center - suggestion_circle_r),
                     (suggestion_window_center + suggestion_circle_r),
                     (suggestion_window_center + suggestion_circle_r)]

suggestion_arrow_start_distance = suggestion_circle_r + 20

right_arrow = [(suggestion_window_center + suggestion_arrow_start_distance),
               (suggestion_window_center + suggestion_arrow_start_distance),
               (suggestion_window_center + suggestion_arrow_start_distance),
               (suggestion_window_center - suggestion_arrow_start_distance),
               suggestion_window_dim,
               suggestion_window_center]

top_arrow = [(suggestion_window_center - suggestion_arrow_start_distance),
             (suggestion_window_center - suggestion_arrow_start_distance),
             (suggestion_window_center + suggestion_arrow_start_distance),
             (suggestion_window_center - suggestion_arrow_start_distance),
             suggestion_window_center,
             0]

bottom_arrow = [(suggestion_window_center - suggestion_arrow_start_distance),
                (suggestion_window_center + suggestion_arrow_start_distance),
                (suggestion_window_center + suggestion_arrow_start_distance),
                (suggestion_window_center + suggestion_arrow_start_distance),
                suggestion_window_center,
                suggestion_window_dim]

left_arrow = [(suggestion_window_center - suggestion_arrow_start_distance),
              (suggestion_window_center - suggestion_arrow_start_distance),
              (suggestion_window_center - suggestion_arrow_start_distance),
              (suggestion_window_center + suggestion_arrow_start_distance),
              0,
              suggestion_window_center]

confidence_window_width = 500
confidence_window_height = 300


def handle_keypress(event):
    print(event.char)


def render(canvas, observation):
    img_raw = Image.fromarray(observation)
    img_raw = img_raw.resize(
        (round(img_raw.size[0] * 3), round(img_raw.size[1] * 3)))
    img = ImageTk.PhotoImage(image=img_raw)
    canvas.create_image(350, 350, anchor="center", image=img)


if __name__ == '__main__':
    # play()

    env = gym.make('MsPacman-v0')
    observation = env.reset()

    window = tk.Tk()
    left_frame = tk.Frame(master=window, borderwidth=10)
    right_frame = tk.Frame(master=window, borderwidth=10)

    game_canvas = tk.Canvas(master=left_frame, bd=2, bg='black',
                            width=game_width, height=game_height)
    checkbox = tk.Checkbutton(master=right_frame, text='Autopilot')
    ai_suggestion_canvas = tk.Canvas(master=right_frame, bd=2,
                                     bg='black', width=suggestion_window_dim, height=suggestion_window_dim)
    ai_confidence_label = tk.Label(master=right_frame, text='AI confidence')
    ai_confidence_canvas = tk.Canvas(master=right_frame, bd=2,
                                     bg='black', width=confidence_window_width, height=confidence_window_height)

    game_canvas.pack()
    checkbox.pack()
    ai_suggestion_canvas.pack()
    ai_confidence_label.pack()
    ai_confidence_canvas.pack()

    left_frame.grid(row=0, column=0)
    right_frame.grid(row=0, column=1)

    window.bind("<Key>", handle_keypress)

    # TODO game rendering
    img_raw = Image.fromarray(observation)
    img_raw = img_raw.resize(
        (round(img_raw.size[0] * 3), round(img_raw.size[1] * 3)))
    img = ImageTk.PhotoImage(image=img_raw)
    game_canvas.create_image(350, 350, anchor="center", image=img)
    # window.after(game, 30, render, game, observation) #FIXME fuckin funkcie, need help

    # TODO ai suggestions
    ai_suggestion_canvas.create_polygon(
        right_arrow, outline='green', fill='yellow', width=3)
    ai_suggestion_canvas.create_polygon(
        top_arrow, outline='green', fill='yellow', width=3)
    ai_suggestion_canvas.create_polygon(
        bottom_arrow, outline='green', fill='yellow', width=3)
    ai_suggestion_canvas.create_polygon(
        left_arrow, outline='green', fill='yellow', width=3)
    ai_suggestion_canvas.create_oval(
        suggestion_circle, outline='green', fill='yellow', width=3)

    # TODO confidence plot

    window.mainloop()