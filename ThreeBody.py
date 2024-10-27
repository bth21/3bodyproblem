#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 23:15:47 2024

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
        
    def acceleration(self, other, softing_parameter):
        """
        Function to find acceleration of one mass due to other mass
        
        acceleration = G*m*(r2-r1)/|r2-r1|^3
        
        Also check if object escapes
        """
        
        body2 = other[0]
        body3 = other[1]
  
        relative_position1 = body2.position - self.position 
        relative_position2 = body3.position - self.position
        acceleration1 = G * body2.mass * relative_position1 / ((np.linalg.norm(relative_position1)+softing_parameter) ** 3)
        acceleration2 = G * body3.mass * relative_position2 / ((np.linalg.norm(relative_position2)+softing_parameter) ** 3)
        acceleration = acceleration1 + acceleration2
        return acceleration
    
    def next_acceleration(self, other, next_position, softing_parameter):
        """
        Function to find acceleration at the next time interval
        """
        
        body2 = other[0]
        body3 = other[1]
        relative_position1 = body2.position - next_position
        relative_position2 = body3.position - next_position
        acceleration1 = G * body2.mass * relative_position1 / ((np.linalg.norm(relative_position1)+softing_parameter) ** 3)
        acceleration2 = G * body3.mass * relative_position2 / ((np.linalg.norm(relative_position2)+softing_parameter) ** 3)
        acceleration = acceleration1 + acceleration2
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
        softing_parameter = self.adaptive_softing(other)
        collision = False # Set default collision parameter to False
        if self.handle_collision(other): # Check if collision occurs
            collision = True # Update collision parameter to True if collision detected
        acceleration = self.acceleration(other, softing_parameter) # Find acceleration of mass
        
        next_position = self.position + self.velocity * self.dt + 0.5 * acceleration * self.dt ** 2 # Find position at next time interval
        next_acceleration = self.next_acceleration(other, next_position, softing_parameter)     # Find acceleration at next time interval
        next_velocity = self.velocity + 0.5 * (acceleration + next_acceleration) * self.dt # Find velocity at next time interval
        self.update_velocity(next_velocity) # Update velocity
        self.update_positions(next_position) # Update positions
        
        return next_position, collision # Returns the next position and whether a collision occurs
    
    def handle_collision(self, other):
        """
        Function to check if a collision occurs - 
        if distance between objects is less than sum of radii then collision
        """
        body2 = other[0]
        body3 = other[1]
        radius1 = body2.radius
        radius2 = body3.radius
        radius = [radius1, radius2]
        radius = max(radius)
      
        relative_position1 = np.linalg.norm(body2.position - self.position)
        relative_position2 = np.linalg.norm(body3.position - self.position)
        if relative_position1 <= (self.radius + radius) or relative_position2 <= (self.radius + radius):
            print('Collision detected')
            return True
        return False
    
    def adaptive_softing(self, other):
        """
        Calculates an adaptive softening parameter based on the distance to nearby bodies.
        Softening fades smoothly based on a cubic function.
        """
        body2 = other[0]
        body3 = other[1]
        relative_position1 = np.linalg.norm(body2.position - self.position)
        relative_position2 = np.linalg.norm(body3.position - self.position)
        
        # Minimum distance for applying full softening
        softening_threshold = (self.radius + min(body2.radius, body3.radius)) * 2
        # Calculate distance-based softening parameter
        closest_distance = min(relative_position1, relative_position2)
        
        if closest_distance <= softening_threshold:
            # Smooth cubic fade-out for softening
            softening_factor = (1 - (closest_distance / softening_threshold) ** 3)
            softening_parameter = 1e4 * softening_factor
        else:
            softening_parameter = 1e-3  # Very small softening parameter at large distances
        
        return 0
    
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
        self.position3 = [bodies[2].position]
            
    def main_loop(self):
        """
        Function to run through and find the positions of each body as the system progresses.
        Also plots each position creating animaiton
        """
        positions = np.concatenate((self.position1 + self.position2 + self.position3)) # Find appropriate figure size
        min_pos = min(positions)
        max_pos = max(positions)
        min_pos = min([min_pos, -max_pos]) # Figure size found from initial max abs position value

        # Plot setup for animation
        fig, ax = pl.subplots()
        
        # Create Circle patches with the specified radii for actual size
        body1_patch = patches.Circle((self.position1[0][0], self.position1[0][1]), self.bodies[0].radius, color='black', label="Body 1")
        body2_patch = patches.Circle((self.position2[0][0], self.position2[0][1]), self.bodies[1].radius, color='red', label="Body 2")
        body3_patch = patches.Circle((self.position3[0][0], self.position3[0][1]), self.bodies[2].radius, color='blue', label="Body 3")
        ax.add_patch(body1_patch)
        ax.add_patch(body2_patch)
        ax.add_patch(body3_patch)

        # Trajectory lines for both bodies
        trajectory1, = ax.plot([], [], '-', alpha=0.5, color = 'black')
        trajectory2, = ax.plot([], [], '-', alpha=0.5, color = 'red')
        trajectory3, = ax.plot([], [], '-', alpha=0.5, color = 'blue')
        
        pl.legend()
        
        total_mass = bodies[0].mass + bodies[1].mass + bodies[2].mass
        
        # Find the positions of each object at each time interval between 0 and the max simulation time
        while self.current_time < self.total_time:
            
            next_position1, collision1 = self.bodies[0].verlet_integration(self.bodies[1:])
            next_position2, collision2 = self.bodies[1].verlet_integration((self.bodies[:1] + self.bodies[2:]))
            next_position3, collision3 = self.bodies[2].verlet_integration(self.bodies[:2])
            
            centre_of_mass = (np.array(self.position1[-1]) * bodies[0].mass + np.array(self.position2[-1]) * bodies[1].mass + np.array(self.position3[-1]) * bodies[2].mass)/total_mass
          
            
            # Adjust axes limits to move with heavier object
            if self.bodies[0].mass >= max([self.bodies[1].mass, self.bodies[2].mass]):
                centre = self.bodies[0].position
            elif self.bodies[1].mass > max([self.bodies[0].mass, self.bodies[2].mass]):
                centre = self.bodies[1].position
            elif self.bodies[2].mass > max([self.bodies[0].mass, self.bodies[1].mass]):
                centre = self.bodies[2].position
            margin = 3    
            #ax.set_xlim(centre[0] + min_pos * margin, centre[0] - min_pos * margin)
            #ax.set_ylim(centre[1] + min_pos * margin, centre[1] - min_pos * margin)
            ax.set_xlim(centre_of_mass[0] + min_pos * margin, centre_of_mass[0] - min_pos * margin)
            ax.set_ylim(centre_of_mass[1] + min_pos * margin, centre_of_mass[1] - min_pos * margin)
            
            
            # End loop is collision occurs
            if collision1 or collision2 or collision3:
                print('Collision detected')
                break
            
            self.position1.append(next_position1)
            self.position2.append(next_position2)
            self.position3.append(next_position3)
        
            # Update plot elements
            body1_patch.center = next_position1[:2]  # Update the center of the Circle for body1
            body2_patch.center = next_position2[:2]  # Update the center of the Circle for body2
            body3_patch.center = next_position3[:2]  # Update the center of the Circle for body3
            trajectory1.set_data([p[0] for p in self.position1], [p[1] for p in self.position1])
            trajectory2.set_data([p[0] for p in self.position2], [p[1] for p in self.position2])
            trajectory3.set_data([p[0] for p in self.position3], [p[1] for p in self.position3])
            
            pl.pause(0.01)  # Pause to create animation effect
            
            self.current_time += self.bodies[0].dt
        
        pl.show()

'''
# Figure 8 parameters
position1 = [9.7e5, -2.43e5]
position2 = [-9.7e5, 2.43e5]
position3 = [0, 0]
body1 = Body(position=position1, velocity=[4.662e4, 4.324e4], mass=1.5e26, radius=1e5, dt=0.1)
body2 = Body(position=position2, velocity=[4.662e4, 4.324e4], mass=1.5e26, radius=1e5, dt=0.1)
body3 = Body(position=position3, velocity=[-9.324e4, -8.647e4], mass=1.5e26, radius=1e5, dt=0.1)
bodies = [body1, body2, body3]
Simulation(bodies, 500).main_loop()
'''

'''
# Circular parameters (semi-stable)
factor1 = 2
position1 = [1e5*factor1, 0]
position2 = [-5e4*factor1, 8.66e4*factor1]  # 60 degrees from position1
position3 = [-5e4*factor1, -8.66e4*factor1]  # 60 degrees from position2
factor = 2.24041
body1 = Body(position=position1, velocity=[0, 2.4e4*factor], mass=1.5e25, radius=1e4, dt=0.01)
body2 = Body(position=position2, velocity=[-2.08e4*factor, -1.2e4*factor], mass=1.5e25, radius=1e4, dt=0.01)
body3 = Body(position=position3, velocity=[2.08e4*factor, -1.2e4*factor], mass=1.5e25, radius=1e4, dt=0.01)
bodies = [body1, body2, body3]
Simulation(bodies, 50).main_loop()
'''

# Figure 8 parameters
position1 = [9.7e5, -2.43e5]
position2 = [-9.7e5, 2.43e5]
position3 = [0, 0]
body1 = Body(position=position1, velocity=[5e4, 4e4], mass=1.5e26, radius=5e4, dt=0.1)
body2 = Body(position=position2, velocity=[5e4, 5e4], mass=1.5e26, radius=5e4, dt=0.1)
body3 = Body(position=position3, velocity=[-9e4, -8e4], mass=1.5e26, radius=5e4, dt=0.1)
bodies = [body1, body2, body3]
Simulation(bodies, 500).main_loop()




