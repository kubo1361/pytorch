from matplotlib import style
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gym_wrapper import transform_observation
import torch
import gym
import numpy as np
from networks import networkV2
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib


# Global parameters
game_width = 700
game_height = 700

suggestion_window_dim = 300
suggestion_window_center = suggestion_window_dim / 2
suggestion_circle_r = suggestion_window_dim / 8
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

up_arrow = [(suggestion_window_center - suggestion_arrow_start_distance),
            (suggestion_window_center - suggestion_arrow_start_distance),
            (suggestion_window_center + suggestion_arrow_start_distance),
            (suggestion_window_center - suggestion_arrow_start_distance),
            suggestion_window_center,
            0]

down_arrow = [(suggestion_window_center - suggestion_arrow_start_distance),
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

matplotlib.use("TkAgg")
style.use("dark_background")

FONT = ("Verdana", 12)

f = Figure(figsize=(5, 5), dpi=60)
a = f.add_subplot(111)

path = 'models/finalfinal/final_2_a2c.pt'
actions = 5


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

        # extract highest probability
        highest_prob = probs.max()

        # sort probabilities
        actions = probs.multinomial(num_samples=1)

        # return highest probability
        return actions[0].item(), highest_prob.item()


class GameCanvas():
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(master=parent, bd=2, bg='black',
                                width=game_width, height=game_height)
        self.canvas.pack()

        self.done = True
        self.env = gym.make('MsPacman-v0')

    def transform_image(self, observation):
        img_raw = Image.fromarray(observation)
        img_raw = img_raw.resize(
            (round(img_raw.size[0] * 3), round(img_raw.size[1] * 3)))
        img = ImageTk.PhotoImage(image=img_raw)
        return img

    def reset(self):
        observation = self.env.reset()
        self.done = False

        image = self.transform_image(observation)
        agent_img = transform_observation(observation)

        return image, agent_img

    def step(self, action):
        observation, _, self.done, _ = self.env.step(action)

        agent_img = transform_observation(observation)
        image = self.transform_image(observation)

        return image, agent_img

    def game_render(self, action):
        done = self.done
        if(self.done):
            img, agent_img = self.reset()
        else:
            img, agent_img = self.step(action)

        self.parent.img = img
        self.canvas.create_image(
            350, 350, anchor="center", image=img)

        return agent_img, done


class SuggestionCanvas():
    def __init__(self, parent):
        self.autopilot = True
        self.autopilot_label = tk.Label(
            master=parent, text='Autopilot')

        self.canvas = tk.Canvas(master=parent, bd=2, bg='black',
                                width=suggestion_window_dim, height=suggestion_window_dim)

        self.autopilot_label.pack()
        self.canvas.pack()

        self.last_visible = 0
        self.arrows = []

        self.arrows.append(self.canvas.create_oval(
            suggestion_circle, outline='yellow', fill='green', width=3,
            state=tk.HIDDEN))

        for type in (up_arrow, right_arrow, left_arrow, down_arrow):
            self.arrows.append(self.canvas.create_polygon(
                type, outline='yellow', fill='green', width=3,
                state=tk.HIDDEN))

    def switch_autopilot(self, switch):
        self.autopilot = switch
        if(self.autopilot):
            self.autopilot_label.pack()
        else:
            self.autopilot_label.forget()

    def suggestion_render(self, action):
        self.canvas.itemconfigure(
            self.arrows[self.last_visible],
            state=tk.HIDDEN)

        if(action == 0):
            self.canvas.itemconfigure(
                self.arrows[0], state=tk.NORMAL)
            self.last_visible = 0
            return

        if(action == 1):
            self.canvas.itemconfigure(
                self.arrows[1], state=tk.NORMAL)
            self.last_visible = 1
            return

        if(action == 2):
            self.canvas.itemconfigure(
                self.arrows[2], state=tk.NORMAL)
            self.last_visible = 2
            return

        if(action == 3):
            self.canvas.itemconfigure(
                self.arrows[3], state=tk.NORMAL)
            self.last_visible = 3
            return

        if(action == 4):
            self.canvas.itemconfigure(
                self.arrows[4], state=tk.NORMAL)
            self.last_visible = 4
            return


class ConfidencePlotCanvas():
    def __init__(self, parent):
        self.canvas = FigureCanvasTkAgg(f, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.canvas._tkcanvas.pack()
        self.x = np.zeros(10)
        self.y = np.zeros(10)

    def update(self, prob):
        next = self.x[-1] + 1
        self.x[0] = next
        self.x = np.roll(self.x, -1)

        self.y[0] = prob
        self.y = np.roll(self.y, -1)

        a.clear()
        a.set_ylim(0, 1)
        a.plot(self.x, self.y)
        self.canvas.draw()

    def reset(self):
        self.x = np.zeros(10)
        self.y = np.zeros(10)


class App():
    def __init__(self):
        self.window = tk.Tk()

        tk.Tk.wm_title(self.window, "Pacman assisted")
        self.window.resizable(False, False)

        self.left_container = tk.Frame(self.window)
        self.right_container = tk.Frame(self.window)
        self.left_container.grid(row=0, column=0)
        self.right_container.grid(row=0, column=1)

        self.game_canvas = GameCanvas(self.left_container)
        self.suggestion_canvas = SuggestionCanvas(self.right_container)
        self.confidence_canvas = ConfidencePlotCanvas(self.right_container)
        self.agent = Agent(networkV2(actions))
        self.agent.load_model(path)

        self.window.bind('<Right>', lambda event: self.right_pressed())
        self.window.bind('<Left>', lambda event: self.left_pressed())
        self.window.bind('<Up>', lambda event: self.up_pressed())
        self.window.bind('<Down>', lambda event: self.down_pressed())
        self.window.bind('<space>', lambda event: self.space_pressed())

        self.stack = np.zeros((4, 80, 80), dtype=np.float32)
        self.ai_control = True
        self.action = 0
        self.ai_prob = 0
        self.ai_action = 0
        self.counter = 0
        self.render()

    def right_pressed(self):
        self.action = 2

    def left_pressed(self):
        self.action = 3

    def up_pressed(self):
        self.action = 1

    def down_pressed(self):
        self.action = 4

    def space_pressed(self):
        self.ai_control = not self.ai_control
        self.suggestion_canvas.switch_autopilot(self.ai_control)
        print("AI control: ", self.ai_control)

    def render(self):
        agent_img, reset = self.game_canvas.game_render(
            (self.action, self.ai_action)[self.ai_control])
        self.suggestion_canvas.suggestion_render(
            self.ai_action)

        if not reset:
            self.confidence_canvas.update(self.ai_prob)
        else:
            self.confidence_canvas.reset()

        if(self.counter % 4 == 0):
            self.stack[:-1] = self.stack[1:]
            self.stack[-1] = agent_img

            self.ai_action, self.ai_prob = self.agent.choose_action(
                torch.from_numpy(self.stack))

        self.counter += 1
        self.window.after(1, lambda: self.render())


if __name__ == '__main__':
    app = App()
    app.window.mainloop()
