## Deep Q-Learning Based Treasure Hunt Game

## Overview
This project implements a Deep Q-Learning algorithm in a simulated treasure hunt environment where an AI agent (the pirate) is trained to navigate through a maze to find a treasure. The primary goal is to demonstrate the effectiveness of Deep Q-Learning in training agents to perform in environments with defined states, actions, and rewards.

## Primary Goal
The main objective of this project is to employ a Deep Q-Learning algorithm in training an AI agent (pirate) to proficiently navigate through a complex maze and locate the treasure efficiently. Through this, we aim to showcase the capability of reinforcement learning in enabling machines to perform complex tasks in dynamic environments.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.7 or later is installed.
- You have installed the required Python libraries and dependencies.

## Installation
Clone the repository to a local directory:
```bash
git clone https://github.com/your-repo-link.git
```

## Usage
## Training the Agent
To start training the agent, run the following script:
```
python train.py
```

## Running the Simulation
```
python simulate.py
```

## Code Examples
Below is a simplified version of the deep Q-learning algorithm used for our agent.
```
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Initialize agent with the state and action size
agent = DQNAgent(state_size=4, action_size=4)

# Train the agent
done = False
batch_size = 32

for e in range(num_episodes):
    # Reset state at the beginning of each game
    state = env.reset()
    
    # time_t represents each frame of the game
    # Our goal is to keep the agent running till the game is done
    for time_t in range(5000):
        # Decide action
        action = agent.act(state)
        
        # Advance the game to the next frame based on the action.
        next_state, reward, done, _ = env.step(action)
        
        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)
        
        # make next_state the new current state for the next frame.
        state = next_state
        
        # done becomes True when the game ends
        if done:
            # print the score and break out of the loop
            print("episode: {}/{}, score: {}".format(e, num_episodes, time_t))
            break

    # Train the agent with the experience of the episode
    agent.replay(batch_size)

# Save the final model
agent.save("dqn.h5")
```

## What do computer scientists do and why does it matter?
Computer scientists solve complex problems through computation. They design algorithms, solve problems, and create efficient systems. The field matters because it drives innovation and efficiency in multiple sectors, from healthcare and finance to education and entertainment. In this project, implementing a Deep Q-Learning algorithm is a practical application of computer science. It showcases how theoretical understanding of algorithms can solve real-world problems, like navigating a complex environment.

## How do I approach a problem as a computer scientist?
As a computer scientist, problem-solving is usually approached with a systematic methodology:
Understand the problem: Break it down into smaller, manageable parts.
Plan a solution: This might involve creating or adapting algorithms, deciding on suitable data structures, and more.
Implement the solution: Write code and integrate systems while considering efficiency, readability, and scalability.
Test the solution: Ensure it works as expected under a variety of conditions, and fix any issues that arise.
Iterate: Based on testing, feedback, and new requirements, revise the approach and the code.

## Ethical Responsibilities:
What are my ethical responsibilities to the end user and the organization?
Privacy and Security: Protecting user data and ensuring the security of systems to prevent data breaches.
Accuracy and Fairness: Ensuring the algorithms are accurate and do not perpetuate biases, especially in AI systems. In the context of the project, it's ensuring the AI behaves as expected and does not develop strategies that might be considered cheating or unfair.
Transparency: Being honest about the capabilities of the software, potential issues, data usage policies, and more.
Sustainability: Writing efficient code that's not wasteful in terms of computational resources, which has broader environmental impacts.
Beneficence and Non-Maleficence: Striving to ensure that the technologies developed are beneficial to society and do not cause harm.

