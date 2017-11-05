# Map file for the self driving car

# Importing the library
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
# Basically this is the brain of the AI in the car --> from the Ai.py file
# Dqn stands for deep q network
from ai import DQN

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0 # the total number of points in the last drawing
length = 0 # the length of the last drawing

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = DQN(5,3,0.95) # 5 sensors, 3 actions, gama = 0.9
action2rotation = [0,20,-20] # action = 0 => no rotation, action = 1 => rotate 20 degres, action = 2 => rotate -20 degres
last_reward = 0 # initializing the last reward
scores = [] # initializing the mean score curve (sliding window of the rewards) with respect to time


# Initializing the map
first_update = True # using this trick to initialize the map only once
def init():
    global sand # sand is an array that has as many cells as our graphic interface has pixels. Each cell has a one if there is sand, 0 otherwise.
    global goal_x # x-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global goal_y # y-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global first_update
    sand = np.zeros((longueur,largeur)) # initializing the sand array with only zeros
    goal_x = 20 # the goal to reach is at the upper left of the map (the x-coordinate is 20 and not 0 because the car gets bad reward if it touches the wall)
    goal_y = largeur - 20 # the goal to reach is at the upper left of the map (y-coordinate)
    first_update = False # trick to initialize the map only once

# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):

    angle=NumericProperty(0) # Here we are initializing an angle variable
                             # Basically this is the angle of the car (angle between the x-axis of the map and the axis of the direciton of the car)

    rotation=NumericProperty(0) # This initializes the last roation of the car (Recall that after playing an action of goig straight, left, or right
                                # The car does a rotation of either 0, 20 or -20 degrees)
    velocity_x=NumericProperty(0) # Initializing the x-coordinate of the velocity Vector
    velocity_y=NumericProperty(0) # Initializing the y-coordinate of the velocity Vector
    velocity = ReferenceListProperty(velocity_x, velocity_y) # velocity vector

    # The next few lines of code are going to describe the sensors on the car
    # Note that there are 3 sensors:
    # Sensor 1: Basically the one that allows the car to sense anything in the forward direction
    # Sensor 2: Allows the car to sense anything 30 degrees to the left of it
    # Sensor 3: Allows the car to sense anything 30 degrees to the right of it
    # Please note that all sensors have an x and y coordinate

    sensor1_x = NumericProperty(0) # initializing the x-coordinate of the first sensor (the one that looks forward)
    sensor1_y = NumericProperty(0) # initializing the y-coordinate of the first sensor (the one that looks forward)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y) # first sensor vector

    sensor2_x = NumericProperty(0) # initializing the x-coordinate of the second sensor (the one that looks 30 degrees to the left)
    sensor2_y = NumericProperty(0) # initializing the y-coordinate of the second sensor (the one that looks 30 degrees to the left)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y) # second sensor vector

    sensor3_x = NumericProperty(0) # initializing the x-coordinate of the third sensor (the one that looks 30 degrees to the right)
    sensor3_y = NumericProperty(0) # initializing the y-coordinate of the third sensor (the one that looks 30 degrees to the right)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y) # third sensor vector

    # The next few lines are going to be populated with the singnal variable that in essense will receive the value obtained by the
    # the sensor variables
    signal1 = NumericProperty(0) # initializing the signal received by sensor 1
    signal2 = NumericProperty(0) # initializing the signal received by sensor 2
    signal3 = NumericProperty(0) # initializing the signal received by sensor 3

    # The next block of code will be for the move function which as the name suggest allows for the car to move in a specific directiion
    # Note that the car must take in a rotation value

    def move(self, rotation):
        self.pos=Vector(*self.velocity) + self.pos # Basically what this does is that it takes the car's velocity and its current position and adds them
                                                   # to get a new position
        self.rotation=rotation # This line of code gets the rotation of the car which is going to be determined below
        self.angle=self.angle + self.rotation # This line of code gets the rotation of the car based on its current value added with the rotation vector

        # Once the car's angle is updated, we must also update the position angles of the sensors on the car
        # One thing that you will notice is that there is a Vector(30,0) from which the rotation from the car's angle is being made
        # This is there because 30 is the distance between the car and the sensor. Note that both are stationary objects
        self.sensor1=Vector(30,0).rotate(self.angle)+self.pos # This will update the position of sensor 1
        self.sensor2=Vector(30,0).rotate((self.angle+30)%360) + self.pos# This will update the positio of sensor 2 which you have to remember looks at things sensed from a 30 degrees to the left of the car
        self.sensor3=Vector(30,0).rotate((self.angle-30)%360) + self.pos# This will update the position of sensor whcih you have to remember looks at the things sensed from a 30 degrees to the right of the car

        # After there are values in the sensors, we then have to update the signals as well
        # Please note that the signals look at the density of sand obtained from the sand array in blocks of 20 by 20
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400. # getting the signal received by sensor 1 (density of sand around sensor 1)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400. # getting the signal received by sensor 2 (density of sand around sensor 2)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400. # getting the signal received by sensor 3 (density of sand around sensor 3)

        # Now the next thing we have to do is penalize the car if the sensors move into the walled bounderies of the map
        # We can do this  by setting up explicity boundaries using if conditionals and then changing the singal strengths to 1 indicating the highest density of sand  in leau of a penalty
        if self.sensor1_x > longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10: # if sensor 1 is out of the map (the car is facing one edge of the map)
            self.signal1 = 1 # sensor 1 detects full sand
        if self.sensor2_x > longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10: # if sensor 2 is out of the map (the car is facing one edge of the map)
            self.signal2 = 1 # sensor 2 detects full sand
        if self.sensor3_x > longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10: # if sensor 3 is out of the map (the car is facing one edge of the map)
            self.signal3 = 1 # sensor 3 detects full sand

class Ball1(Widget): # sensor 1 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball2(Widget): # sensor 2 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball3(Widget): # sensor 3 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass

# Creating the game class

# The next block of code will be for the game class which will be the thing the AI car plays to go from the airport to the downtown and vice versa
class Game(Widget):

    car=ObjectProperty(None) # This gets the car object from our kivy file
    ball1=ObjectProperty(None) # This gets the sensor1 object from our kivy file
    ball2 = ObjectProperty(None) # This gets the sensor 2 object from our kivy file
    ball3 = ObjectProperty(None) # This gets the sensor 3 object from our kivy file

    def serve_car(self):
        self.car.center = self.center # the car will start at the center of the map
        self.car.velocity = Vector(6, 0) # the car will start to go horizontally to the right with a speed of 6


    # This next block of code is going to be the big update function that that will update everything at each point in time t when
    # the car reaches a new state (i.e. getting new signals from the car)
    def update(self, dt):

        global brain # specifying the global variables (the brain of the car, that is our AI)
        global last_reward # specifying the global variables (the last reward)
        global scores # specifying the global variables (the means of the rewards) the scores are synonomous to the rewards obtained by the car
        global last_distance # specifying the global variables (the last distance from the car to the goal)
        global goal_x # specifying the global variables (x-coordinate of the goal)
        global goal_y # specifying the global variables (y-coordinate of the goal)
        global longueur # specifying the global variables (width of the map)
        global largeur # specifying the global variables (height of the map)

        longueur = self.width # width of the map (horizontal edge)
        largeur = self.height # height of the map (vertical edge)
        if (first_update): # trick to initialize the map only once
            init() # Remember that we populated this function above

        # The next portion of code pertains to the actual updates itself

        # The next two lines of code look at the delta of distance between the goal state and the cars current postiion
        xx=goal_x-self.car.x
        yy=goal_y-self.car.y

        # The following line of code gets the orientation of the car
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180. # This gets the direction of the car with respect to the goal state position
                                                                     # If the direction=0 then this means that the car is heading perfectly in the direction toward the goal state

        # The following line of code gets the most recent signal from the car
        # Note that there is both a positive orientation and a negative orientation, this is because in order for the car to explore the map fully, it needs both directions of the orientation
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] # our input state vector, composed of the three signals received by the three sensors, plus the orientation and -orientation
        action = brain.update(last_reward, last_signal) # playing the action from our ai (the object brain of the dqn class), formulates an action response by the last reward as well as the last signal
        scores.append(brain.score()) # appending the score (mean of the last 100 rewards to the reward window)
        rotation = action2rotation[action] # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        self.car.move(rotation) # moving the car according to this last rotation angle
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2) # getting the new pythagorean distance between the car and the goal right after the car moved
        self.ball1.pos = self.car.sensor1 # updating the position of the first sensor (ball1) right after the car moved
        self.ball2.pos = self.car.sensor2 # updating the position of the second sensor (ball2) right after the car moved
        self.ball3.pos = self.car.sensor3 # updating the position of the third sensor (ball3) right after the car moved


        # The next few blocks of code will have to do with the penalties with regards to both the velocity of the car as well as the reward it will receive when it is or isn't moving on sand

        # First we will create a conditional for when the car is on sand:
        if (sand[int(self.car.x),int(self.car.y)] > 0): # This just looks at the cars position in the sand array and checks if the value of the cell is 1 which indicates that sand is there in that pixel
            self.car.velocity = Vector(1, 0).rotate(self.car.angle) # basically the first thing that happens is that the velocity of the car is slowed down from its normal velocity of 6 to 1
            last_reward=-1 # Here we want to set the harshest negative reward onto the car by setting the value of the last_reward to -1 o the car moves off the sand

        # Then we want to observe what happens when the car is not on the sand
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle) # When the car is not on sand then it will go at a normal speed of 6
            last_reward= -0.2 # Here we want to introduce a small negative reward on the car every time it isn't at the goal state

            # Here we set a sub conditional that will reward the car with a  small positive reward if the car is getting closer to the goal state
            if (distance < last_distance):
                last_reward=0.1 # We give the car a small reward if the car gets closer to the goal state

        # The next block of code will deisgnate a harsh negative reward when the car is in the edge case and is at the edge of the map
        if (self.car.x < 10): # if the car is in the left edge of the frame
            self.car.x = 10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if (self.car.x > self.width-10): # if the car is in the right edge of the frame
            self.car.x = self.width-10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if (self.car.y < 10): # if the car is in the bottom edge of the frame
            self.car.y = 10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if (self.car.y > self.height-10): # if the car is in the upper edge of the frame
            self.car.y = self.height-10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1

        # Now we want to handle for the case when the car actually manages to reach the goal
        if (distance < 100):
            goal_x=self.width - goal_x # What this does is that this changes the goal state to the bottom right corner of the map which is what we will call the down town (it will flip flop back to the upper left corner of the current goal state is the downtown)
                                       # Note that this is for the x-coordinate
            goal_y=self.height - goal_y # What this does is that this changes the goal state to the bottom right corner of the map which is what we will call the down town (it will flip flop back to the upper left corner of the current goal state is the downtown)
                                        # Note that this is for the y-coordinate
        # Now we have to update the last_distance of the car
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()
