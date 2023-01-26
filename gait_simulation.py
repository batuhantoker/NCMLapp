import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import animation

def run_simulation(height, cycles,theta_max,torso_width,simulation, filename):
    pelvis_length = torso_width/100
    height = height / 100
    pendulum_length = height*0.53


    def draw_body(body_left,body_right):
        body_dim = height * 0.4
        x = [body_left[0], body_right[0], body_left[0], body_right[0]]
        y = [body_left[1], body_right[1], body_left[1], body_right[1]]
        z = [body_left[2], body_right[2], body_left[2]+body_dim, body_right[2]+body_dim]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts,linewidth=10,edgecolor="k",facecolor = "red",rasterized=False))
        r = 0.01
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:30j]
        x = (body_left[0]+ body_right[0])/2+(np.cos(u) * np.sin(v))*0.4
        y = (body_left[1]+ body_right[1])/2+((np.sin(u) * np.sin(v))*0.06)
        z = body_left[2]+body_dim+.28 + np.cos(v) *0.2
        ax.plot_surface(x, y, z,color='r')

    def draw_line(pos1, pos2):
        for i in range(3):
            xs = [pos1[0], pos2[0]]
            ys = [pos1[1], pos2[1]]
            zs = [pos1[2], pos2[2]]
            # Plot contour curves
            line1, = ax.plot(xs, ys, zs,'k', linewidth=5)
        return line1

    #time.sleep(3) # wait 3 seconds

    def motion_generator(cycles):
        pos_pendulum_left = np.array([0, 0, 0])
        pos_pendulum_right = np.array([0, pelvis_length, 0])
        body_left = np.array([0, 0, pendulum_length])
        body_right = np.array([0, pelvis_length, pendulum_length])
        body_left = np.vstack([body_left, body_left])
        body_right = np.vstack([body_right, body_right])
        pos_pendulum_left = np.vstack([pos_pendulum_left, pos_pendulum_left])
        pos_pendulum_right = np.vstack([pos_pendulum_right, pos_pendulum_right])
        theta_1, theta_2, cycle_definition_left, cycle_definition_right, at_change = motion_planner(cycles)
        pend_prev = [pos_pendulum_right[0], pos_pendulum_left[0]]
        body_prev = [body_right[0], body_left[0]]
        for index , cycle in enumerate(cycle_definition_left):
            if at_change[index]==1:
                pend_prev = [pos_pendulum_right[index],pos_pendulum_left[index]]
            if cycle == 1:
                pos_pendulum_new_left = pend_prev[1]

                body_new_left = [pos_pendulum_new_left[0] + pendulum_length * np.sin(np.deg2rad(theta_1[index])),
                                       0,
                                 pendulum_length *(np.cos(np.deg2rad(theta_2[index]))) ]
                body_new_right = [body_new_left[0],pelvis_length,body_new_left[2]] #[body_prev[0, 0] + pendulum_length * np.sin(np.deg2rad(theta[index])),
                                # body_prev[0, 1],
                                 #pendulum_length * np.cos(np.deg2rad(theta[index]))]
                pos_pendulum_new_right = [body_new_right[0] + pendulum_length * np.sin(np.deg2rad(theta_2[index])),
                                          body_new_right[1],
                                          body_new_right[2] - pendulum_length * np.cos(np.deg2rad(theta_2[index]))]
                body_left = np.vstack([body_left, body_new_left])
                body_right = np.vstack([body_right, body_new_right])
                pos_pendulum_left =  np.vstack([pos_pendulum_left, pos_pendulum_new_left])
                pos_pendulum_right = np.vstack([pos_pendulum_right, pos_pendulum_new_right])
            if cycle == -1:
                pos_pendulum_new_right = pend_prev[0]

                body_new_right = [pos_pendulum_new_right[0] + pendulum_length * np.sin(np.deg2rad(theta_1[index])),
                                 pelvis_length,
                                  pendulum_length *(np.cos(np.deg2rad(theta_1[index])))]
                body_new_left = [body_new_right[0], 0, body_new_right[2]]  # [body_prev[0, 0] + pendulum_length * np.sin(np.deg2rad(theta[index])),
                # body_prev[0, 1],
                # pendulum_length * np.cos(np.deg2rad(theta[index]))]
                pos_pendulum_new_left = [body_new_right[0] + pendulum_length * np.sin(np.deg2rad(theta_1[index])),
                                          0,
                                          body_new_right[2] - pendulum_length * np.cos(np.deg2rad(theta_1[index]))]
                body_left = np.vstack([body_left, body_new_left])
                body_right = np.vstack([body_right, body_new_right])
                pos_pendulum_left = np.vstack([pos_pendulum_left, pos_pendulum_new_left])
                pos_pendulum_right = np.vstack([pos_pendulum_right, pos_pendulum_new_right])




        return pos_pendulum_left, body_left,pos_pendulum_right,body_right,theta_1,theta_2

    def motion_planner(cycles):

        theta_1 = np.linspace(0, theta_max , data_points)
        theta_2 = np.linspace(0,theta_max,  data_points)
        at_change = [0]  * (data_points)
        cycle_definition_left =  [1]  * (data_points)
        cycle_definition_right = [-1]  * (data_points)
        for i in range(1,cycles*2+1):
            if i % 2 ==0:
                theta_1 = np.append(theta_1,np.linspace(-theta_max,theta_max,  data_points))
                theta_2 = np.append(theta_2,np.linspace( -theta_max, theta_max, data_points))
                cycle_definition_left = np.append(cycle_definition_left, [1] * data_points)
                cycle_definition_right = np.append(cycle_definition_right, [-1] * data_points)
                at_change = np.append(at_change,np.append([1], [0] * (data_points-1)))
            else:
                theta_1 = np.append(theta_1, np.linspace(-theta_max, theta_max, data_points))
                theta_2 = np.append(theta_2, np.linspace(-theta_max,theta_max, data_points))
                cycle_definition_left = np.append(cycle_definition_left, [-1] * data_points)
                cycle_definition_right = np.append(cycle_definition_right, [1] * data_points)
                at_change = np.append(at_change, np.append([1], [0] * (data_points-1)))
        return theta_1,theta_2,cycle_definition_left,cycle_definition_right,at_change





    data_points = 20
    pos_pendulum_left, body_left, pos_pendulum_right, body_right,theta_1,theta_2 = motion_generator(cycles)
    def update(frame):
        i = frame
        ax.clear()


        ax.axes.set_xlim3d(left=body_right[i, 0] - 2, right=body_right[i, 0] + 2)
        ax.axes.set_ylim3d(bottom=-.10, top=pelvis_length + .10)
        ax.axes.set_zlim3d(bottom=0, top=height+.20)
        # ax.azim = -90
        # ax.dist = 20
        # ax.elev = 0
        ax.scatter(pos_pendulum_left[i, 0], pos_pendulum_left[i, 1], pos_pendulum_left[i, 2], c='b', marker='o')
        ax.scatter(pos_pendulum_right[i, 0], pos_pendulum_right[i, 1], pos_pendulum_right[0, 2], c='g', marker='o')
        ax.scatter(body_left[i, 0], body_left[i, 1], body_left[i, 2], c='r', marker='o')
        ax.scatter(body_right[i, 0], body_right[i, 1], body_right[i, 2], c='k', marker='o')
        draw_line(pos_pendulum_left[i,:], body_left[i,:])
        draw_line(body_left[i,:], body_right[i,:])
        draw_line(body_right[i,:], pos_pendulum_right[i,:])
        draw_body(body_left[i,:], body_right[i,:])

        ax.set_box_aspect((60, 40, 50))
        ax.set_xlabel('(m)', fontsize=10)
        ax.set_ylabel('(m)', fontsize=10)

        ax.plot(body_right[:i,0],body_right[:i,1],body_right[:i,2], linewidth=.5, dashes=[5, 3], c='k')

        ax.plot(body_left[:i, 0],body_left[:i, 1], body_left[:i, 2], linewidth=.5, dashes=[5, 3], c='r')
        ax.plot(pos_pendulum_right[:i, 0], pos_pendulum_right[:i, 1], pos_pendulum_right[:i, 2], linewidth=.5, dashes=[5, 3], c='g')
        ax.plot(pos_pendulum_left[:i, 0], pos_pendulum_left[:i, 1], pos_pendulum_left[:i, 2], linewidth=.5, dashes=[5, 3], c='b')
        # ax2.clear()
        # ax2.plot(body_right[:i, 2], linewidth=.5, dashes=[5, 3], label = 'Body, z axis')
        # ax2.legend()
        # ax2.set_xlabel('Simulation time (samples)', fontsize=10)
        # ax2.set_ylabel('Value (m)', fontsize=10)
        # canvas2.draw()
        # canvas2.flush_events()

        canvas.draw()
        canvas.flush_events()


    if simulation == 1:
        anim = animation.FuncAnimation(fig, update, frames=100)
        anim.save(filename, writer='imagemagick', fps=30)
    else :
        for i in range(pos_pendulum_left.shape[0]):  # range(len(pos_pendulum_left)):
            update(i)
root = tk.Tk()
root.title("Gait Simulation")
root.geometry("900x900")
cycle_label = tk.Label(root, text="Number of Cycles:")
cycle_label.grid(row=0,  column=0, sticky='e')
cycle_entry = tk.Entry(root)
cycle_entry.grid(row=0,  column=1, sticky='w')
cycle_entry.insert(0, "4")


height_label = tk.Label(root, text="Height (cm):")
height_label.grid(row=1,  column=0, sticky='e')
height_entry = tk.Entry(root)
height_entry.grid(row=1,  column=1, sticky='w')
height_entry.insert(0, "183")
theta_label = tk.Label(root, text="Theta (degree):")
theta_label.grid(row=2,  column=0, sticky='e')
theta_entry = tk.Entry(root)
theta_entry.grid(row=2,  column=1, sticky='w')
theta_entry.insert(0, "20")

pelvis_label = tk.Label(root, text="Torso width (cm):")
pelvis_label.grid(row=3,  column=0, sticky='e')
pelvis_entry = tk.Entry(root)
pelvis_entry.grid(row=3,  column=1, sticky='w')
pelvis_entry.insert(0, "15")

run_button = tk.Button(root, text="Run Simulation", command=lambda: run_simulation(float(height_entry.get()), int(cycle_entry.get()),int(theta_entry.get()),int(pelvis_entry.get()),0,0))
run_button.grid(row=4, column=1, sticky='w')
animation_label = tk.Label(root, text="Animation file name")
animation_label.grid(row=5,  column=0, sticky='e')
animation_entry = tk.Entry(root)
animation_entry.grid(row=5,  column=1, sticky='w')
animation_entry.insert(0, "animation.gif")
save_button = tk.Button(root, text="Save animation", command=lambda: run_simulation(float(height_entry.get()), int(cycle_entry.get()),int(theta_entry.get()),int(pelvis_entry.get()),1,animation_entry.get()))
save_button.grid(row=6, column=1, sticky='w')
fig_frame = tk.Frame(root)
fig_frame.grid(row=7,column=0,columnspan=2)
# Create a placeholder figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
canvas = FigureCanvasTkAgg(fig, master=fig_frame)

canvas.draw()
canvas.get_tk_widget().grid(row=0,  column=1)
# # Create a placeholder figure
# fig2 = plt.figure()
# ax2=plt.axes()
# canvas2 = FigureCanvasTkAgg(fig2, master=fig_frame)
# canvas2.draw()
# canvas2.get_tk_widget().grid(row=0,  column=2)


root.mainloop()
