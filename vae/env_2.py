import pygame
import numpy as np
import math
import time
from gym.spaces.box import Box
import matplotlib.pyplot as plt
import pickle
import gzip

f =gzip.open('./screenshot_data2002003_3.gzip','wb')

class Reacher_for2:
    def __init__(self, screen_size=1000, link_lengths = [200, 140], joint_angles=[0, 0], target_pos = [669,430], render=False):
        # Global variables
        self.screen_size = screen_size
        self.link_lengths = link_lengths
        self.joint_angles = joint_angles
        self.num_actions=len(link_lengths)  # equals to number of joints - 1
        self.num_observations= 2*(self.num_actions+2)
        self.L = 8 # distance from target to get reward 2
        self.action_space=Box(-100,100, [self.num_actions])
        self.observation_space=Box(-1000,1000, [2*(self.num_actions+2)])

        # The main entry point
        self.render=render
        if self.render == True:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Reacher")
        else:
            pass
        self.is_running = 1
        # self.target_pos=[self.screen_size/4, self.screen_size/4]
        self.target_pos=target_pos

        self.steps=0
        self.max_episode_steps=50
        self.near_goal_range=0.5
        self.change_goal=0
        self.change_goal_episodes=10



    # Function to compute the transformation matrix between two frames
    def compute_trans_mat(self, angle, length):
        cos_theta = math.cos(math.radians(angle))
        sin_theta = math.sin(math.radians(angle))
        dx = -length * sin_theta
        dy = length * cos_theta
        T = np.array([[cos_theta, -sin_theta, dx], [sin_theta, cos_theta, dy], [0, 0, 1]])
        return T


    # Function to draw the current state of the world
    def draw_current_state(self, ):
        # First link in world coordinates
        T_01 = self.compute_trans_mat(self.joint_angles[0], self.link_lengths[0])
        origin_1 = np.dot(T_01, np.array([0, 0, 1]))
        p0 = [0, 0]
        p1 = [origin_1[0], -origin_1[1]]  # the - is because the y-axis is opposite in world and image coordinates
        # Second link in world coordinates
        T_12 = self.compute_trans_mat(self.joint_angles[1], self.link_lengths[1])
        origin_2 = np.dot(T_01, np.dot(T_12, np.array([0, 0, 1])))
        p2 = [origin_2[0], -origin_2[1]]  # the - is because the y-axis is opposite in world and image coordinates
        # Third link in world coordinates
        # T_23 = self.compute_trans_mat(self.joint_angles[2], self.link_lengths[2])
        # origin_3 = np.dot(T_01, np.dot(T_12, np.dot(T_23, np.array([0, 0, 1]))))
        # p3 = [origin_3[0], -origin_3[1]]  # the - is because the y-axis is opposite in world and image coordinates
        # Compute the screen coordinates
        # print(p0,p1,p2,p3)
        p0_u = int(0.5 * self.screen_size + p0[0])
        p0_v = int(0.5 * self.screen_size + p0[1])
        p1_u = int(0.5 * self.screen_size + p1[0])
        p1_v = int(0.5 * self.screen_size + p1[1])
        p2_u = int(0.5 * self.screen_size + p2[0])
        p2_v = int(0.5 * self.screen_size + p2[1])
        # p3_u = int(0.5 * self.screen_size + p3[0])
        # p3_v = int(0.5 * self.screen_size + p3[1])
        # Draw
        if self.render == True:
            self.screen.fill((0, 0, 0))
            line_width=20  # origin 5
            circle_size=20 # origin 10
            pygame.draw.line(self.screen, (255, 255, 255), [p0_u, p0_v], [p1_u, p1_v], line_width)
            pygame.draw.line(self.screen, (255, 0, 255), [p1_u, p1_v], [p2_u, p2_v], line_width)
            # pygame.draw.line(self.screen, (255, 255, 255), [p2_u, p2_v], [p3_u, p3_v], 5)
            pygame.draw.circle(self.screen, (100, 255, 0), [p0_u, p0_v], circle_size)
            pygame.draw.circle(self.screen, (200, 0, 255), [p1_u, p1_v], circle_size)
            pygame.draw.circle(self.screen, (255, 0, 0), [p2_u, p2_v], circle_size)
            # pygame.draw.circle(self.screen, (255, 0, 0), [p3_u, p3_v], 10)
            
            pygame.draw.circle(self.screen, (255, 255, 0), np.array(self.target_pos).astype(int), circle_size*2)
            # Flip the display buffers to show the current rendering
            pygame.display.flip()
            # time.sleep(1)

            ''' screenshot the image '''
            # pygame.image.save(self.screen, './screen.png')
            array_screen = pygame.surfarray.array3d(self.screen) # 3d array pygame.surface (self.screen)
            red_array_screen=pygame.surfarray.pixels_red(self.screen) # 2d array from red pixel of pygame.surface (self.screen)
            # print(array_screen.shape)
            downsampling_rate=5 # downsmaple the screen shot, origin 1000*1000*3
            CHANNELS=['rgb', 'red'][1]
            if CHANNELS == 'red': # 2d, need to expand 1 dim
                downsampled_array_screen=np.expand_dims(red_array_screen[::downsampling_rate,::downsampling_rate,], axis=-1)
            elif CHANNELS == 'rgb':
                downsampled_array_screen=array_screen[::downsampling_rate,::downsampling_rate,]
            pickle.dump(downsampled_array_screen, f)  # generate screenshot data
            # plt.imshow(array_screen[::downsampling_rate,::downsampling_rate,])
            # plt.show()
            
        else:
            pass
        return [p0_u,p0_v,p1_u,p1_v,p2_u,p2_v], np.array([downsampled_array_screen])
    
    def reset(self,):
        self.steps=0
        self.joint_angles = np.array([0.1,0.1])*180.0/np.pi
        if self.render == True:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Reacher")
        else:
            pass
        self.is_running = 1

        # reset the target position for learning across tasks
        self.change_goal+=1
        if self.change_goal > self.change_goal_episodes:
            self.change_goal=0
            range_pose=0.3
            target_pos=range_pose*np.random.rand(2) + [0.5,0.5]
            self.target_pos=target_pos*self.screen_size

        pos_set, screenshot=self.draw_current_state()
        
        return screenshot

    def step(self,action, sparse_reward=False):    
        # Get events and check if the user has closed the window
        if self.render == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = 0
                    break
        else:
            pass
        # Change the joint angles (the increment is in degrees)
        change=np.random.uniform(-1,1,size=3)
        self.joint_angles[0] += action[0][0]
        self.joint_angles[1] += action[0][1]
        # self.joint_angles[2] += action[2]
        # Draw the robot in its new state
        # print(action)
        pos_set, screenshot=self.draw_current_state()
        if sparse_reward:
            '''sparse reward'''
            distance2goal = np.sqrt((pos_set[4]-self.target_pos[0])**2+(pos_set[5]-self.target_pos[1])**2)
            
            if distance2goal < self.L:
                reward = 20
            else:
                reward = -1
            return screenshot, np.array([reward]), np.array([False]), distance2goal
        
        else:    
            '''dense reward'''
            reward_0=100.0
            reward = reward_0 / (np.sqrt((pos_set[4]-self.target_pos[0])**2+(pos_set[5]-self.target_pos[1])**2)+1)

            # time.sleep(0.3)
            # 10 dim return
            return screenshot, np.array([reward]), np.array([False])


if __name__ == "__main__":
    reacher=Reacher_for2(render=True)  # 2-joint reacher
    num_episodes=500
    num_steps=20
    action_range=50.0


    epi=0
    while epi<num_episodes:
        print(epi)
        epi+=1
        step=0
        reacher.reset()
        while step<num_steps:
            step+=1
            action=np.random.uniform(-action_range,action_range,size=3)
            reacher.step([action])

    f.close()

    


    
