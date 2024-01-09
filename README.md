<h2><b>Simulation of a Two Arm Actuator Using Reinforcement Learning</b></h2>


<h2> About </h2>
<hr/>

<img src="https://media.discordapp.net/attachments/782728868179607603/1185070191424581702/Screenshot_2023-12-14_at_11.05.47_PM.png?ex=658e459f&is=657bd09f&hm=5c407973395e83d203756f0da9ca4e6ab0db2b61973098f012c0086745cfa5a4&=&format=webp&quality=lossless&width=1344&height=960">
Image taken from <a href="https://www.researchgate.net/figure/Free-body-diagram-of-the-robot-arm_fig11_268437501"> researchgate.net. </a>

The purpose of this project is to recreate a model seen in the paper <b>"Actuator Trajectory Planning for UAVs with
Overhead Manipulator using Reinforcement Learning" </b> which was written by Hazim Alzorgan, Abolfazl Razi and Ata Jahangir Moshayed.
The paper describes an application of reinforcement learning on a virtual drone, where various kinematics are taken into account
in simulating it. In addition the drone has a two arm actuator on top with a tip known as
an end effector. Its goal is to detect a viable base path and to follow a target path with the tip of the end effector.

This project utilizes methods from the paper such as inverse kinematic equations, and 
Q-Learning formulas to recreate a more bare-bones version of what was presented in the paper.

<h2> Running It </h2>
<hr/>

To run my program you will need to install the required libraries including Gym. On installing Gym,
place the PathEvn file into the `python3.11 > site-packages > gym > envs > classic_control` folder. 
A common error I've encountered on new installs is the custom environment not registering with Gym,
I fixed this by duplicating my working Gym folder and replacing the broken one. I have provided this working 
Gym folder in the repo. 

Here's an explanation of the files:
- /gym/ - You can use your own install of gym, but this one is here incase you encounter errors.
- base_path.npy - These are the base path coordinates the drone body follows.
- ee_path.npy - These are the coordinates of the target path the end effector should follow.
- PathEnv.py - This is the custom environment used to model the arm and world.
- q-table.csv - This is a table containing the q-values of after a 10,000 episode run in Q mode.
- README.md - See README for more details.
- TestRunQ.py - This is the Q-Learning algorithm which tests and renders the program.
- TestRunSARSA.py - This is a modified version of TestRunQ.py which implements SARSA. 

Note: There is a commented section of code near the bottom of TestRunQ.py and TestRunSARSA.py which allow for rendering the simulation in pygame. There is also a section of code toward the bottom of TestRunQ.py and TestRunSARSA.py which can be uncommented to see the statstics of a run after a certain episode.


<h2> The Stack </h2>
<hr/>
The main library used to enable reinforcement learning was OpenAI's Gym. This 
allowed for a custom environment to be created and used on top of existing libraries in Gym
to create a custom reinforcement learning project. 

To create a graphical representation of the environment, I used PyGame to 
render and grab the data from my environment. Other packages like, numpy, and 
pandas were used as well. 

One of the biggest challenges I faced early on was figuring out how the 
kinematic equations would fit. I learned after a while that a confusion with
what functions returned degrees and radians led me to go in circles for 
a couple hours. Getting the inverse kinematic equations to function was a hassle 
early on. Another challenge was figuring out how the observation and action space 
would work. A specific challenge with the action space was deciding whether it should
be a discrete set from the get-go, or if it should be continous and quantized later. 
The process of sorting out where those functions belonged also took until now to realize fully. 


<h2> Equations </h2>
<hr/>
<img src="https://cdn.discordapp.com/attachments/782728868179607603/1184743438948765707/Screenshot_2023-12-14_at_1.27.08_AM.png?ex=658d154f&is=657aa04f&hm=8a3b4f60a4be43b7e4b1fe2d887bbf043d7c3dcee1c39914363a08ed2f170643&"> 

These equations taken from the paper listed above, derive the inverse kinematics of the 
two arm actuator. These two equations show the respective angles of the root and middle joint
of the arm. 

The basis of the Q Learning algorithm follows this pseudocode. 

<img src="https://zitaoshen.rbind.io/img/Q-learning/q-learning.png">

A subtle difference exists between Q-Learning and SARSA but it can make a big impact.

<img src="https://vinitsarode.weebly.com/uploads/1/0/3/7/103702208/screenshot-from-2018-07-08-02-28-03_orig.png">
<a href="https://vinitsarode.weebly.com/blogs/sarsa-vs-q-learning"></a>

In Q-learning, the Q-value of the current state-action pair is updated based on the maximum Q-value of the next state. 
But in SARSA, the Q-value of the current state-action pair is updated based on the Q-value of the next state and the 
action taken in that state. 

Q-learning follows an off-policy approach. It learns the optimal policy while following a 
different, exploratory policy (usually epsilon-greedy). This means that Q-learning estimates the value of the best action 
even if the agent does not necessarily choose that action during exploration. Whereas SARSA follows an on-policy approach. 
It learns the value of the policy it is currently using. This means that SARSA estimates the value of the action that 
the agent would actually take in the next state.

<h2> Approach and Challenges </h2>
<hr/>

Delving into reinforcement learning was quite the learning curve. OpenAI Gym stood out as the most approachable library 
for me to grasp, making it the natural choice to start with. Creating a custom environment, though, turned out to be a 
laborious task, as at first it was incredibly unstable. Pygame posed 
its own set of challenges. As figuring out the scaling and coordinates of objects 
was not a straightforward as it could have been. When I first started working on the visuals for 
this project I had planned on using PyBullet. However given the 2D nature of the sim, and the challenging
logistics of creating and rigging a model, I found it impractical given time constraints. 

When I began work on creating the custom environment I imagined a 1 dimensional version of the gridworld problem that 
is always used in reinforcement learning examples. The two most important things to set up were the observation
space and the action space. I knew that each state was a discrete location on a path, and that 
the next position was guaranteed, but I didn't know at first how I would keep track of the angle values of each position.

This is where Gym's spaces came in handy. As I could define the observation space as a tuple, where the first value
is a discrete space of integers from 0 to 99, and the second value could be the angular measurements as another tuple
this time defined as a Box space between the upper and lower limits. 

<img src="https://cdn.discordapp.com/attachments/782728868179607603/1185000946976161792/Screenshot_2023-12-14_at_12.38.16_PM.png?ex=658e0522&is=657b9022&hm=09fb5f864f815678270058534065831930623912255b0df3c50a61d292530dfa&">

For the action space, I defined it as a continuous Box space going between -3 degrees and positive 3
degrees. I decided to have my actions represent different values of torque movements that could be made. With this case, both arms would
be moving at once, unless a random action was selected otherwise with a 0. 

This project tested both Q-Learning and SARSA, and those details will be discussed below in findings.


<h2> Findings </h2>
<hr/>

This program implemented a Q-Learning and a SARSA learning algorithm.
The results of the Q learning algorithm yielded an average reward across 1000 episodes of 
<b> -11.034.</b> The SARSA run across 1000 episodes yielded a reward of <b>-10.940.</b> This is hard to see visually
but it does represent at least a small improvement over Q learning. The last photo shows a run over 5000
episodes in SARSA mode. This time the average reward was lowered to <b>-10.872.</b>

<img src="https://media.discordapp.net/attachments/782728868179607603/1185008345950715987/Screenshot_2023-12-14_at_6.59.47_PM.png?ex=658e0c06&is=657b9706&hm=fd144fe21fcb936e009ff14f52b05f9af78c63cc65441f11c438e589cb393078&=&format=webp&quality=lossless&width=1370&height=1028">
<img src="https://media.discordapp.net/attachments/782728868179607603/1185012626175049829/Screenshot_2023-12-14_at_7.16.51_PM.png?ex=658e1002&is=657b9b02&hm=6f5aba424154d449af8b2d65a242e577fe8301e576e6d6ef24e37b0099162d92&=&format=webp&quality=lossless&width=1370&height=1028">
<img src="https://media.discordapp.net/attachments/782728868179607603/1185015687534620713/Screenshot_2023-12-14_at_7.28.35_PM.png?ex=658e12dc&is=657b9ddc&hm=f5e0189d71c26f7e55ba4ae6b8c813622460bea619c6ed648521f8220bf69cb2&=&format=webp&quality=lossless&width=1378&height=1028">


https://github.com/yeetbruises/Two-Arm-Actuator-AI-RL-Sim/assets/61666396/bf462356-c3a3-4124-9026-32a9a60f53e1

