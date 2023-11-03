import numpy as np
import os

def read_trajectory_from_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    traj_dir = os.path.join(parent_dir, 'traj')
    file_path = os.path.join(traj_dir, file_name)
    
    trajectory = []
    number_of_steps = 0

    try:
        with open(file_path, 'r') as file:
            for line in file:
                number_of_steps += 1
                linedata = [float(number) for number in line.split()]
                trajectory.append(linedata)
    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return np.array(trajectory), number_of_steps

def ref_cb(line_to_read, QUADROTOR_N, QUADROTOR_NY, number_of_steps, trajectory, acados_in):
    if QUADROTOR_N + line_to_read + 1 <= number_of_steps:  # All ref points within the file
        for i in range(QUADROTOR_N + 1):  # Fill all horizon with file data
            for j in range(QUADROTOR_NY + 1):
                acados_in.yref[i][j] = trajectory[i + line_to_read][j]
    elif line_to_read < number_of_steps:  # Part of ref points within the file
        for i in range(number_of_steps - line_to_read):  # Fill part of horizon with file data
            for j in range(QUADROTOR_NY + 1):
                acados_in.yref[i][j] = trajectory[i + line_to_read][j]
        for i in range(number_of_steps - line_to_read, QUADROTOR_N + 1):  # Fill the rest horizon with the last point
            for j in range(QUADROTOR_NY + 1):
                acados_in.yref[i][j] = trajectory[number_of_steps - 1][j]
    else:  # none of ref points within the file
        for i in range(QUADROTOR_N + 1):  # Fill all horizon with the last point
            for j in range(QUADROTOR_NY + 1):
                acados_in.yref[i][j] = trajectory[number_of_steps - 1][j]


if __name__ == '__main__':

    file_name = "circle.txt"
    trajectory, number_of_steps = read_trajectory_from_file(file_name)
    print(number_of_steps)