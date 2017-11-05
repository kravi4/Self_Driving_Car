# Note that this AI files defines the function for the decision making process of the car using deep q learning
# It's basically the brain of the car

# In sum, this is the AI for the Self Driving Car

# Importing of the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn # module for neural networks
import torch.nn.functional as F # contains the different functions when implementing a neural network
import torch.optim as optim # optimizers that will be for the back propagating for stochastic gradient decent
import torch.autograd as autograd
from torch.autograd import Variable # basically something that will be able store the tensor and the gradient

# Creating the architecture of the Neural Network
class Network(nn.Module):

    # init function will take in two other parameters besides self. The input size will be the input states
    # which will be a of size 5, three for the sensors and 2 for the postive and negative orientations
    # The other parameter is the actions that the that the car can take. These are the output neurons of the
    # neural network which correspond to the the three actions the car can take (going straight, left, or right)
    def __init__(self, input_size, nb_actions):
        super(Network, self).__init__() # basically a way to use all the functionalitie sof the nn.module class
        self.input_size=input_size
        self.nb_actions=nb_actions

        # The next variable will be the full connections between the layers of our neural network
        # Since we are only going to have one hidden layer in our neural network, we are going to have 2 full connections
        # There will be one fool connection between our input layer and our hidden layer and then another full connection between our hidden layer and our output layer

        # Creates the full connections between the input layer and the hidden layer,
        # Full connection implies that all the neurons of the one layer are connection to all the neurons of the corresponding connected layer
        # The nn.Linear() function builds the connections between the layers
        # In the nn.Linear() function we have to specify 2 parameters, the first parameter is the number of number of neruons of the first input feature and the 2nd parameter is the number of neurons of the output feature
        # Note that there is a bias parameter as well that is initialized to true by default, this specifies that there are weights assigned to the neurons which is important for back propagation
        # In the case of the first parameter will be the value of the input_size parameter passed above
        # In this case we can specify the number of neurons of the hidden layer to be anything, but for the sake of testing lets try out 30
        self.fc1=nn.Linear(input_size, 30)

        # Now we will initialize the full connections between our hidden layer and the output layer
        # Note that here the input parameter will be the number of nodes in the hidden layer which is 30 and the second parameter will output_size which is the nb actions
        self.fc2=nn.Linear(30,30)
        self.fc3=nn.Linear(30,30)
        self.fc4=nn.Linear(30, nb_actions)



        # Please note that we can have many more hidden layers with varying neuron input size so long as there is a path between the input and the output
        # In fact, more hidden layers might actually improve the overall perfomance of the car

    # The following function is the forward function that will activate the neurons using forward propagation
    # This fucntion will return a q
    def forward(self, state):
         # first thing that we are going to do is to activate the hidden neurons
         # We will then use a recitfier function (since we don't intend the decision making process to be binary) to activate the hidden layer
         # basically relu takes in an input parameter which is the hidden layer which is obtained by passing in the state into out fc1 full connection to
         # which passes the initial state into the input neurons and populates the hidden layer
         h1 = F.relu(self.fc1(state))
         h2 =F.relu(self.fc2(h1))
         h3 =F.relu(self.fc3(h2))

         # This will generate the q_values by taking the activated hidden layer and then passing that into out fc2 full connection to derive the result
         q_values=self.fc4(h3)
         return q_values

         # In the event that we created more hidden layers, we have to continuall activate those hidden layers using the relu function and then connecting everything to the output
         # An example of this would be that if we had 3 hidden layers (assuming that the full connection variables were already described in the init function):
         # h1=F.relu(self.fc1(state))
         # h2=F.relu(self.fc2(h1))
         # h3=F.rele(self.fc3(h2))
         # output_q=self.fc4(h3)

# We will now implement the class for experience replay
# This will designate a markov decision process on up to 100 states in the past
# Then we will randomly sample any portion of the memory in the replay
# This class will have the init function, the push function that ensure that the class never has more than 100,000 memories, and then a sample function that will take a sample out of the memories
class replay_memory(object):
    # Note that capacity will be set to 100,000 in the map file that will store a capacity of 100,000 memories or transitions in the meory of events
    def __init__(self, capacity):
        self.capacity=capacity
        # The memory should be a list of the last 100 memories
        # Note at the beginning there are no memories in the memory list
        self.memory=[]

    # Now we will create a push function that pushes a memory onto the memory list and also enures that the capacity of the memory list never goes above 100,000
    # This function should take in self and also an event in which it can put into its memory list
    # Note that event will be in the form of a tuple where the tuple looks something like this (last_state, new_state, last_action, last_reward)
    def push(self, event):
        # Lets append te event to the memory list
        self.memory.append(event)
        # Now we have to set a limit on the memory list to ensure that it bounded by the capacity
        # once we go above the capacity, we want to delete the first element in the memory in the memory list or the earliest memory
        # Alternatively we probably could have used a queue here instead
        if (len(self.memory)>self.capacity):
            del self.memory[0]

    # The next function is called sample and will take random samples from our memory list
    # This function takes in two parameters. One is self and the other is the batch size or the total random sampling size of the memories from the meory list that we randomly sample
    def sample(self, batch_size):
        # The zip function rechapes the list
        # An example of this is as follows imagine list=[(1,2,3) , (4,5,6)] the zip(*list)=[(1,4), (2,3), (5,6)]
        # The zip function is needed to group each value by state, action, and reward
        # random.sample samples the batch_ize from the memory
        sample= zip(*random.sample(self.memory, batch_size))

        # We have to put the samples into a pytorch variable
        # We can use the map function which stores the sampls as a tensor and a gradient
        # The first parameter is a function that will convert the samples to a torch variable
        # The second parameter is what we will execute the funtion on
        return map(lambda x: Variable(torch.cat(x,0)), sample)

# Implementing Deep Q learning
class DQN():
    # The init function will take in the input size the actions as well as the gamma parameter in the Bellman Q equation
    def __init__(self, input_size, nb_actions, discount_gamma):
        self.discount_gamma=discount_gamma

        # The next variable is going to be a slding window of the for teh reward of the last 100 memories
        self.reward_window=[] # In the beginning, however, this has to be zero since there are no memories here

        # This next variable is going to instantiate the neural network that the Q-learning system wil be a applied to
        # ---> We will use the network class that we described above
        self.model=Network(input_size, nb_actions)

        # Now we have to initialize the memory using the replay function
        self.memory=replay_memory(100000)

        # This will optimize the gradient decent
        # WE have to pass in model parameters as well as a learning rate
        # Note that we don't want the learning rate to be so high because we want the AI to take time exploring and learning the system
        # We will set it here as 0.001 but other values could work as well
        self.optimizer=optim.Adam(self.model.parameters(), lr=0.001)

        # Basically here we have to turn the tuple or vector of our current state into a tensor
        self.last_state=torch.Tensor(input_size).unsqueeze(0)
        self.last_action=0
        self.last_reward=0

    # Now we need to define a function that chooses the right action that allows the car to move toward the goal while avoiding the sand
    # The action that we will choose will depend on the output of the neural network
    def select_action(self, state):
        # probs will store the probabilites of the each action as derived by the soft max which will give the highest probability to the action that goes to the state with the highest q-value
        # basically we have to execute a soft max on the output of model's neural network
        # We could just pass in our state since it is in the form of a tensor but since we want to associate the gradient or the weight along with the tensor
        # we can convert the state to a torch variable while setting the volatile parameter to True to maintain the gradient weight
        # After that, we have to multiply this with a termperture value. The closer the termperature is to 0 the less sure the Ai will be in making a decision
        Temperature=100
        probs=F.softmax(self.model(Variable(state, volatile=True))*Temperature)
        # This will give us a random draw from the probs distribution
        action=probs.multinomial()
        return (action.data[0,0])

    # Now we will create a learning function that executes the forward propagation
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # gathers all the outputs from the used actions squeezing and unsqueezing is just to efficintetly use the tensors
        output = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Then we have to take the max of the next states
        next_output=self.model(batch_next_state).detach().max(1)[0]
        # As per the derived bellman equation for Q-Learning, we know that the target value will equal the gamma time the expected output added with the batch reward
        target=self.discount_gamma*next_output+batch_reward
        # Now we have to calculate the error in the prediction
        # We will use termporal difference to for this predictio error determination
        # Basiacally, we want to find the hoover loss between our target and our output
        # We can get the hoover loss by using the F.smooth_l1_loss function
        td_loss=F.smooth_l1_loss(output, target)
        # We re-inialize our optimizer at each iteration of the loop
        self.optimizer.zero_grad()
        # Now we have to do backward propagation from the td_loss to re-adjust the weights
        td_loss.backward(retain_variables=True)
        # Now we have to update the weights by taking the optimizer and then uusing the step function to update the weights
        self.optimizer.step()

    # Now we will have to create  function that updates everything after the AI gets into a new state
    # This update function will take the last reward and the last signal
    def update(self, new_reward, new_signal):
        new_state=torch.Tensor(new_signal).float().unsqueeze(0)
        # We now have to upadate the memory by pushing a new event to the memory altogether
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action=self.select_action(new_state)
        # Now we have to learn from the last 100 memories
        # but in order to do this we have to ensure that we have 100 memories that exist anyway
        if (len(self.memory.memory)>100):
            # Then we want initialze the batch values to a sampling of the memory with a batch sample size of 100
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            # Then we want our network to learn from those batch_next_state
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        # Now we need to update the last_state, last_action, last_reward, and the reward_window and make any neccessary deletes to that if necessary
        self.last_action=action
        self.last_state=new_state
        self.last_reward=new_reward
        self.reward_window.append(new_reward)
        if (len(self.reward_window) > 1000):
            del self.reward_window[0]
        return action


    # Now we will have a function that computes a score on the sldiing window of rewards
    # basically we will compute the mean of all the rewards in the reward window
    def score(self):
        # We have to ensure that the denominator is never 0 so in order to make this happen, we have to add 1 to the length
        return (sum(self.reward_window)/len(self.reward_window)+1)

    # Now let us create a function that can save the specifc brain of an AI in a specific point in time
    def save(self):
        # The model is saved with all the weights as well as the optimizer
        torch.save({
        'state_dict': self.model.state_dict(),
        'optimizer' : self.optimizer.state_dict(),
        }, 'last_brain.pth')

    # Now Let us create a function that will load the weights and the optimizer from the saved last_brain file
    def load(self):
        # First we have to check if the file exists
        if (os.path.isfile('last_brain.pth')==True):
            print("===> LOADING CHECKPOINT......")
            checkpoint = torch.load('last_brain.pth')
            # Now that we have loaded the file, we have to udated both the existing model and the optimizer
            # with the wights from the saved weights and optimizer from the saved file
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Succesfully Loaded")
        else:
            print("There is no saved checkpoint")
