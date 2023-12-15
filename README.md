<h3><b>Simulation of a Two Arm Actuator Using Reinforcement Learning</b></h3>


<h2> About </h2>
<hr/>

<img src="https://www.researchgate.net/profile/Ashraf-Elfasakhany/publication/268437501/figure/fig11/AS:668635869745155@1536426622280/Free-body-diagram-of-the-robot-arm.ppm">
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

There is a section of code toward the bottom of TestRunQ.py and TestRunSARSA.py which can 
be uncommented to see the statstics of a run after a certain episode. 

Examples of graphs that can be plotted using the in-built plotter:

<img src="https://media.discordapp.net/attachments/782728868179607603/1185008345950715987/Screenshot_2023-12-14_at_6.59.47_PM.png?ex=658e0c06&is=657b9706&hm=fd144fe21fcb936e009ff14f52b05f9af78c63cc65441f11c438e589cb393078&=&format=webp&quality=lossless&width=1370&height=1028">
<img src="https://media.discordapp.net/attachments/782728868179607603/1185012626175049829/Screenshot_2023-12-14_at_7.16.51_PM.png?ex=658e1002&is=657b9b02&hm=6f5aba424154d449af8b2d65a242e577fe8301e576e6d6ef24e37b0099162d92&=&format=webp&quality=lossless&width=1370&height=1028">
<img src="https://media.discordapp.net/attachments/782728868179607603/1185015687534620713/Screenshot_2023-12-14_at_7.28.35_PM.png?ex=658e12dc&is=657b9ddc&hm=f5e0189d71c26f7e55ba4ae6b8c813622460bea619c6ed648521f8220bf69cb2&=&format=webp&quality=lossless&width=1378&height=1028">
