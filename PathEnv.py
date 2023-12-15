"""
Vineet Saraf
12/13/2023
CPSC 4420
PathEnv.py
"""
import itertools
import pygame
import math
import numpy as np
import gym
from gym import spaces

"""
Initializing the Pygame
"""
# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1400, 700
ARM_LENGTH_1 = 50
ARM_LENGTH_2 = 50
ARM_LENGTH_3 = 50
JOINT_RADIUS = 5
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
FONT_SIZE = 20

j = 0

clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Variable Control for Arms")
font = pygame.font.Font(None, FONT_SIZE)

running = True
i = 0

"""
Reinforcement learning section
"""

# location on the 1D path
index = 0


# Defines a custom environment for our game to work in
class PathEnv(gym.Env):
    def __init__(self, path):
        super(PathEnv, self).__init__()
        # Load the base path and target path (ee_path) in.
        self.base_path = [[x[0] * 70 + 250, x[1] * 15 + 395] for x in np.load('base_path.npy')]
        self.ee_path = [[x[0] * 83 + 160, x[1] * 40 + 300] for x in np.load('ee_path.npy')]

        # Default starting angles
        self.starting_angles = (-65.0, 19.9)

        # Set to the start of the path, always keep track of current state.
        self.current_position_index = 0
        self.lines = []

        # Gym start_pos req. fulfilled here
        self.start_pos = self.current_position_index  # Starting element of the base_path (not an index)
        self.goal_pos = 99  # Goal position
        self.current_pos: tuple[int, tuple] = (self.start_pos,  # (x, y) This will store the location on the path.
                                               self.starting_angles)  # (q1, q2) This will store the joint angles
        self.num_rows = len(self.ee_path)

        # Angular limits of both arms
        self.ARM1_LOW, self.ARM1_HIGH = -90, -40
        self.ARM2_LOW, self.ARM2_HIGH = 20, 70

        # Create the observation space (Discrete path position, (Tuple of angle limits))
        self.observation_space = spaces.Tuple((
            spaces.Discrete(100),
            spaces.Tuple((
                spaces.Box(low=self.ARM1_LOW, high=self.ARM1_HIGH, shape=(1,)),
                spaces.Box(low=self.ARM2_LOW, high=self.ARM2_HIGH, shape=(1,))
            ))
        ))
        """MODIFY FOR 1 ARM OR 2 ARM EXPERIEMENT"""
        # Describe a continuous space for actions to be derived from, maximum or minimum movement of arms
        self.action_space = gym.spaces.Box(low=np.array([-3.0, -3.0]), high=np.array([3.0, 3.0]), dtype=np.float64)

        # This turns our continuous action space into a an array of a set of discrete actions.
        # [:num_subunits_sqrt] Add only if testing one arm
        self.discretized_actions = self.discretize_box_space()#[:50]

        self.observation_space_n = 100  # Number of observations will be number of places on the path.
        self.action_space_n = len(self.discretized_actions)

        # Initialize Pygame
        pygame.init()

    # Discretify the gradient into n amount of buckets
    def discretize_box_space(self, num_subunits_sqrt=5):
        space = self.action_space
        low = space.low
        high = space.high

        # Generate subunit values for each dimension separately
        subunit_values = [np.linspace(low[i], high[i], num=num_subunits_sqrt) for i in range(len(low))]

        # Create all possible combinations of values
        discretized_actions = list(itertools.product(*subunit_values))

        # Return the array
        return [x for x in discretized_actions]

    def reset(self):
        self.current_pos = (self.start_pos,
                            (np.array([self.starting_angles[0]]).astype(np.float32),
                             np.array([self.starting_angles[1]]).astype(np.float32)))
        self.current_position_index = 0
        screen.fill(WHITE)
        info = {}
        return self.current_pos, info

    """
    Calculate a reward from the difference between the end-effector and target
    """

    def calculate_reward(self, current_position: tuple, new_joint_angles: tuple):
        # Action should be a tuple of two possible arm movements
        arm1_angle, arm2_angle = new_joint_angles
        root_x, root_y = self.base_path[current_position]  # current root position based off of index of base_path

        # Get position of end-effector
        new_ee_x = root_x + ARM_LENGTH_1 * math.cos(math.radians(arm1_angle)) \
                   + ARM_LENGTH_2 * math.cos(math.radians(arm1_angle + arm2_angle))

        new_ee_y = root_y + ARM_LENGTH_1 * math.sin(math.radians(arm1_angle)) \
                   + ARM_LENGTH_2 * math.sin(math.radians(arm1_angle + arm2_angle))
        """
        This reward function is the negative distance from the target point.
        """
        reward = -math.hypot(new_ee_y - self.ee_path[self.current_pos[0]][1],
                             new_ee_x - self.ee_path[self.current_pos[0]][0])
        return reward

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        done = False
        reward = 0  # initialize a reward

        """
        Every step, the agent will move forward in the path, no matter what the action is.
        We will take a random action from our action space and apply it to the game.
        This will happen in the RL part, so here we assume we are given the random action.
        The action takes the form of a given tuple of torques.
        """
        new_pos = self.current_pos
        self.current_position_index += 1

        # Check if the new position is valid
        if 100 > self.current_position_index >= 0:
            # Go to the next spot in our base_path, carry over our current angle joints
            new_pos = (self.current_position_index, (np.array([self.current_pos[1][0]])[0].astype(np.float32),
                                                     np.array([self.current_pos[1][1]])[0].astype(np.float32)))
        else:
            done = True;
            return self.current_pos, reward, done, {}

        # Grab our current arm angles
        curr_angle_arm1 = new_pos[1][0]
        curr_angle_arm2 = new_pos[1][1]

        # Mapping input action to torques for joint 1 and joint 2
        torque_joint1 = action[0]  # represents torque for joint 1
        torque_joint2 = action[1]  # represents torque for joint 2

        # What if the torque produces an angle outside of what we are allowing?
        # Apply the new angle and calculate a reward
        curr_angle_arm1 += float(torque_joint1);
        curr_angle_arm2 += float(torque_joint2)
        reward = self.calculate_reward(new_pos[0], (curr_angle_arm1, curr_angle_arm2))

        # Have we reached the end of the path?
        if np.array_equal(new_pos[0], self.goal_pos):
            # Once we get to the end of the path we will just set it 1.0 by default.
            reward = 10.0
            done = True

        self.current_pos = new_pos

        return self.current_pos, reward, done, {}

    def render(self, e, i, r, mode="human"):
        screen.fill(WHITE)
        base_path, ee_path = self.base_path, self.ee_path

        # Draw the base path
        spacing = 0
        for p in base_path:
            pygame.draw.circle(screen, (255, 0, 0), (int(p[0]), int(p[1])), 3)
            spacing += 8

        # Draw the ee path
        spacing = 0
        for a in ee_path:
            pygame.draw.circle(screen, (0, 255, 0), (int(a[0]), int(a[1])), 3)
            spacing += 8

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get current joint angles
        q_1 = self.current_pos[1][0]
        q_2 = self.current_pos[1][1]
        # q_3 = self.current_pos[1][2] #---#

        # Get root position by location (x, y) of current point on base_path
        ROOT_POS = [self.base_path[self.current_pos[0]][0],
                    self.base_path[self.current_pos[0]][1]]

        # Calculate positions of joints based on q_1 and q_2 angles
        joint1_pos = [
            ROOT_POS[0] + ARM_LENGTH_1 * math.cos(math.radians(q_1)),
            ROOT_POS[1] + ARM_LENGTH_1 * math.sin(math.radians(q_1))
        ]
        joint2_pos = [
            joint1_pos[0] + ARM_LENGTH_2 * math.cos(math.radians(q_1 + q_2)),
            joint1_pos[1] + ARM_LENGTH_2 * math.sin(math.radians(q_1 + q_2))
        ]

        # This if statement prevents floating point errors which cause domain errors. ex. 1.00001 !< 1
        if (((joint2_pos[0] - ROOT_POS[0]) ** 2 + (
                joint2_pos[1] - ROOT_POS[1]) ** 2 - ARM_LENGTH_1 ** 2 - ARM_LENGTH_2 ** 2)
            / (2 * ARM_LENGTH_1 * ARM_LENGTH_2)) > 1:

            q_2 = math.acos(
                2 - (((joint2_pos[0] - ROOT_POS[0]) ** 2
                      + (joint2_pos[1] - ROOT_POS[1]) ** 2
                      - ARM_LENGTH_1 ** 2
                      - ARM_LENGTH_2 ** 2)
                     /
                     (2 * ARM_LENGTH_1 * ARM_LENGTH_2))
            ) * (180 / 3.14159265358929)

        else:
            q_2 = math.acos(
                ((joint2_pos[0] - ROOT_POS[0]) ** 2
                 + (joint2_pos[1] - ROOT_POS[1]) ** 2
                 - ARM_LENGTH_1 ** 2
                 - ARM_LENGTH_2 ** 2)
                /
                (2 * ARM_LENGTH_1 * ARM_LENGTH_2)
            ) * (180 / 3.14159265358929)

        q_1 = (-1) * \
              (
                      (-1) * math.degrees(math.atan((joint2_pos[1] - ROOT_POS[1]) / (joint2_pos[0] - ROOT_POS[0])))
                      +
                      math.degrees(math.atan((ARM_LENGTH_1 * math.sin(math.radians(q_2)))
                                             / (ARM_LENGTH_1 +
                                                ARM_LENGTH_2 * math.cos(math.radians(q_2)))))
              )

        # Reset trace route after each episode
        if self.current_pos[0] == 1 or self.current_pos[0] == self.goal_pos:
            self.lines.clear()

        # Create the database for trace route to draw from
        self.lines.append([joint2_pos[0], joint2_pos[1]])

        # Draw the actuator
        pygame.draw.rect(screen, GREEN,
                         (ROOT_POS[0] - JOINT_RADIUS, ROOT_POS[1] - JOINT_RADIUS, JOINT_RADIUS * 2, JOINT_RADIUS * 2))
        text_surface = pygame.font.Font(None, FONT_SIZE).render("Drone", True, BLACK)
        text_rect = text_surface.get_rect(center=(ROOT_POS[0], ROOT_POS[1]))
        screen.blit(text_surface, text_rect)

        # JOINT 1
        pygame.draw.circle(screen, RED, (int(joint1_pos[0]), int(joint1_pos[1])), JOINT_RADIUS)
        # JOINT 2
        pygame.draw.circle(screen, RED, (int(joint2_pos[0]), int(joint2_pos[1])), JOINT_RADIUS)
        # ARM 1
        pygame.draw.line(screen, RED, ROOT_POS, (int(joint1_pos[0]), int(joint1_pos[1])), 5)
        # ARM 2
        pygame.draw.line(screen, RED, (int(joint1_pos[0]), int(joint1_pos[1])),
                         (int(joint2_pos[0]), int(joint2_pos[1])), 5)

        # Display labels for angles
        # LABEL 1
        label_angle_arm_to_root = font.render(f"Angle 1: {-1 * q_1:.2f} degrees", True, BLACK)
        label_rect1 = label_angle_arm_to_root.get_rect(center=(ROOT_POS[0], ROOT_POS[1] - 30))
        screen.blit(label_angle_arm_to_root, label_rect1)

        # LABEL 2
        label_angle_arm_to_arm1 = font.render(f"Angle 2: {-1 * q_2:.2f} degrees", True, BLACK)
        label_rect2 = label_angle_arm_to_arm1.get_rect(
            center=((joint1_pos[0] + joint2_pos[0]) // 2, (joint1_pos[1] + joint2_pos[1]) // 2 - 30))
        screen.blit(label_angle_arm_to_arm1, label_rect2)

        error = 0
        for im in range(len(self.lines)):
            error += math.hypot((self.lines[im][0] - self.ee_path[im % len(self.ee_path)][0]),
                                (self.lines[im][1] - self.ee_path[im % len(self.ee_path)][1]))

        error = error / len(self.lines)

        ep_surface = font.render(f"Episode: {e}", True, (0, 0, 0))  # Black color (0, 0, 0)
        it_surface = font.render(f"Iteration: {i}", True, (0, 0, 0))  # Black color (0, 0, 0)
        rw_surface = font.render(f"Reward: {r}", True, (0, 0, 0))  # Black color (0, 0, 0)

        ep_text = ep_surface.get_rect()
        ep_text.topleft = (100, 120)

        it_text = it_surface.get_rect()
        it_text.topleft = (100, 175)

        rw_text = rw_surface.get_rect()
        rw_text.topleft = (100, 150)

        screen.blit(ep_surface, ep_text)
        screen.blit(it_surface, it_text)
        screen.blit(rw_surface, rw_text)

        text_surface = font.render(f"Average Error: {error:2f}", True, (0, 0, 0))  # Black color (0, 0, 0)
        text_rect = text_surface.get_rect()
        text_rect.topleft = (100, 100)
        screen.blit(text_surface, text_rect)

        for i in range(len(self.lines) - 1):
            pygame.draw.line(screen, (0, 0, 0), self.lines[i], self.lines[i + 1], 2)

        pygame.display.flip()
