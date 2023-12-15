<h3><b>Simulation of a Two Arm Actuator Using Reinforcement Learning</b></h3>


<h2> About </h2>
<hr/>

<img src="https://media.discordapp.net/attachments/782728868179607603/1185070191424581702/Screenshot_2023-12-14_at_11.05.47_PM.png?ex=658e459f&is=657bd09f&hm=5c407973395e83d203756f0da9ca4e6ab0db2b61973098f012c0086745cfa5a4&=&format=webp&quality=lossless&width=1344&height=960">
Image taken from <a href="https://www.researchgate.net/figure/Free-body-diagram-of-the-robot-arm_fig11_268437501"> researchgate.net. </a>

This project utilizes methods such as inverse kinematics, and 
Q-Learning and SARSA to develop a two arm actuator with the goal of successfully tracing out a target path
in a virtual space.

 
<h2> Documentation </h2>
<hr/>

To run my program you will need to install the required libraries including Gym. On installing Gym,
place the PathEnv.py file into the `python3.11 > site-packages > gym > envs > classic_control` folder. 
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

<b> Note: </b>
There is a commented section of code near the bottom of TestRunQ.py and TestRunSARSA.py
which allow for rendering the simulation in pygame. There is also a section of code toward the bottom of TestRunQ.py and TestRunSARSA.py which can 
be uncommented to see the statstics of a run after a certain episode. 

An example of a graph that can be plotted using the in-built plotter:

<img src="https://media.discordapp.net/attachments/782728868179607603/1185008345950715987/Screenshot_2023-12-14_at_6.59.47_PM.png?ex=658e0c06&is=657b9706&hm=fd144fe21fcb936e009ff14f52b05f9af78c63cc65441f11c438e589cb393078&=&format=webp&quality=lossless&width=1370&height=1028">
