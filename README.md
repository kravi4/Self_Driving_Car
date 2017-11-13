# Self_Driving_Car

Hey all, this is my implementation of a Self Driving Car (on a graphical user
interface). There isn't going to be a whole lot in this read me other than a few
technical specifications as to what you need to have downloaded before you actually
run this program.

First things first, you are going to need to download anaconda for Python 3.6.
That will get you all the basic starter python data science stuff that is
very crucial to the running of this program. Next, you will need to install
pytorch. You can do this very easily by going to you terminal window and typing
in pip install pytorch. If that doesn't work just go to the pytorch website and
download it directly. After that you are going to need to pip install kivy.

With all the installation jazz out of the way, we can now get into the fun stuff
of actually running the code. Basically the program is run through the main file.
So in the terminal window you can just run python3 main.py. Now when you do this
you'll see this black screen show up with a "car" randomly moving on the screen.
You can draw different paths around the GUI and the car will actually learn to go
around them. Note that the objective of the car is to go from the top left corner
of the screen to the bottom right corner of the screen. So whatever path you draw,
the car will try to navigate around this path.

Now with all this being said, I figure a lot of you home boys out there are going
to try and draw the most advanced mazes out there. It is very important to Note
that since this car is learning off of deep q learning, the learning done in the
experience replay is off of 100 randomly sampled memories of a memory bank of
100,000. For more complex mazes drawn, you are going to need to adjust the memory
sampled rate in the ai.py file on line 184 and 186. Then whatever you adjust that
to, you need to divide that number from 100,000 and take the result and place that
into the number in line 195. Other considerations to keep in mind is the learning
rate. Note that the learning rate is set to a 0.001 value. I have set the value this
low in large part because I want the car to explore the surrounding environment.
Drawbacks of this, however, might mean that the car will not follow along the maze
that you draw in. You can get around this by increasing the learning rate on line 134
in the AI.py file.
