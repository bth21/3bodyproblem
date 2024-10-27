#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:19:05 2024

@author: benhale
"""

import numpy as np
import scipy.constants as sp
import pylab as pl
import matplotlib.patches as patches


G = sp.G # Gravitational constant

class Body:
    """
    Class to solve EoM for two bodies interacting gravitationally
    """
    def __init__(self, position, velocity, mass, radius, dt):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.mass = mass
        self.radius = radius
        self.dt = dt
        self.previous_position = self.position - self.velocity * self.dt
        
    def acceleration(self, other):
        """
        Function to find acceleration of one mass due to other mass
        
        acceleration = G*m*(r2-r1)/|r2-r1|^3
        
        Also check if object escapes
        """
        lost = False
        relative_position = other.position - self.position 
        acceleration = G * other.mass * relative_position / (np.linalg.norm(relative_position) ** 3)
        if np.linalg.norm(acceleration) <= 1e-3:
            lost = True
        return acceleration, lost
    
    def next_acceleration(self, other, next_position):
        """
        Function to find acceleration at the next time interval
        """
        relative_position = other.position - next_position
        acceleration = G * other.mass * relative_position / (np.linalg.norm(relative_position) ** 3)
        return acceleration
    
    def update_positions(self, next_position):
        """
        Function to move the positions to the next time step
        """

        self.previous_position = self.position
        self.position = next_position
        
    def update_velocity(self, next_velocity):
        """
        Function to update velocities at next time step
        """
        self.velocity = next_velocity
    
    def verlet_integration(self, other):
        """
        Perform Verlet integration
        Check if a collision occurs and break loop if collision exists
        """
        collision = False # Set default collision parameter to False
        if self.handle_collision(other): # Check if collision occurs
            collision = True # Update collision parameter to True if collision detected
        acceleration, lost = self.acceleration(other) # Find acceleration of mass
        
        next_position = self.position + self.velocity * self.dt + 0.5 * acceleration * self.dt ** 2 # Find position at next time interval
        next_acceleration = self.next_acceleration(other, next_position)     # Find acceleration at next time interval
        next_velocity = self.velocity + 0.5 * (acceleration + next_acceleration) * self.dt # Find velocity at next time interval
        self.update_velocity(next_velocity) # Update velocity
        self.update_positions(next_position) # Update positions
        
        return next_position, collision, lost # Returns the next position and whether a collision occurs
    
    def handle_collision(self, other):
        """
        Function to check if a collision occurs - 
        if distance between objects is less than sum of radii then collision
        """
        relative_position = np.linalg.norm(self.position - other.position) # Distance between masses
        if relative_position <= (self.radius + other.radius):
            print('Collision detected')
            return True
        return False
    
class Simulation:
    """
    Class to run through the simulation and create animation
    """
    def __init__(self, bodies, total_time):
        self.bodies = bodies
        self.total_time = total_time
        self.current_time = 0
        self.position1 = [bodies[0].position]
        self.position2 = [bodies[1].position]
            
    def main_loop(self):
        """
        Function to run through and find the positions of each body as the system progresses.
        Also plots each position creating animaiton
        """
        positions = np.concatenate((self.position1 + self.position2)) # Find appropriate figure size
        min_pos = min(positions)
        max_pos = max(positions)
        min_pos = min([min_pos, -max_pos]) # Figure size found from initial max abs position value

        # Plot setup for animation
        fig, ax = pl.subplots()
        
        # Create Circle patches with the specified radii for actual size
        body1_patch = patches.Circle((self.position1[0][0], self.position1[0][1]), self.bodies[0].radius, color='black', label="Body 1")
        body2_patch = patches.Circle((self.position2[0][0], self.position2[0][1]), self.bodies[1].radius, color='red', label="Body 2")
        ax.add_patch(body1_patch)
        ax.add_patch(body2_patch)

        # Trajectory lines for both bodies
        trajectory1, = ax.plot([], [], '-', alpha=0.5, color = 'black')
        trajectory2, = ax.plot([], [], 'r-', alpha=0.5)
        pl.legend()
        
        # Find the positions of each object at each time interval between 0 and the max simulation time
        while self.current_time < self.total_time:
            
            next_position1, collision1, lost1 = self.bodies[0].verlet_integration(self.bodies[1])
            next_position2, collision2, lost2 = self.bodies[1].verlet_integration(self.bodies[0])
            
            # Adjust axes limits to move with heavier object
            if self.bodies[0].mass > self.bodies[1].mass:
                centre = self.bodies[0].position
            elif self.bodies[1].mass > self.bodies[0].mass:
                centre = self.bodies[1].position
               
            margin = 3    
            ax.set_xlim(centre[0] + min_pos * margin, centre[0] - min_pos * margin)
            ax.set_ylim(centre[1] + min_pos * margin, centre[1] - min_pos * margin)
            

            # End loop is collision occurs
            if collision1 or collision2:
                print('Collision detected')
                break
            
            # End loop if the less massive object escapes
            if lost1 and (self.bodies[0].mass < self.bodies[1].mass):
                print('Object escapes1')
               # break
            elif lost2 and (self.bodies[1].mass < self.bodies[0].mass):
                print('Object escapes2')
               # break
            
            self.position1.append(next_position1)
            self.position2.append(next_position2)
        
            # Update plot elements
            body1_patch.center = next_position1[:2]  # Update the center of the Circle for body1
            body2_patch.center = next_position2[:2]  # Update the center of the Circle for body2
            trajectory1.set_data([p[0] for p in self.position1], [p[1] for p in self.position1])
            trajectory2.set_data([p[0] for p in self.position2], [p[1] for p in self.position2])
            
            pl.pause(0.01)  # Pause to create animation effect
            
            self.current_time += self.bodies[0].dt
        
        pl.show()

#v_esc = np.sqrt(G * 5e30 / 10e6)

position1 = [5e5, 0]
position2 = [-5e5, 0]

body1 = Body(position=position1, velocity=[70000, 50000], mass=5e26, radius=1e5, dt=0.1)
body2 = Body(position=position2, velocity=[20000, 150000], mass=5e25, radius=5e4, dt=0.1)


bodies = [body1, body2]

Simulation(bodies, 100).main_loop()


