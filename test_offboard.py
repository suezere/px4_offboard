import numpy as np
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import TimesyncStatus, VehicleOdometry, VehicleStatus, HoverThrustEstimate
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleRatesSetpoint,TrajectorySetpoint

from acados_settings import acados_settings
from utils import *


Tf = 1        # prediction horizon
N = 20     # number of discretization steps
Ts = Tf / N   # sampling time[s]

T_hover = 2     # hovering time[s]
T_traj = 20.00  # trajectory time[s]

T = T_hover + T_traj  # total simulation time

# constants
g = 9.81     # m/s^2

# measurement noise bool
noisy_measurement = False

# input noise bool
noisy_input = False

# extended kalman filter bool
# extended_kalman_filter = False

# generate circulare trajectory with velocties
traj_with_vel = False

# use a single reference point
ref_point = True

# import trajectory with positions and velocities and inputs
import_trajectory = False

# bool to save measurements and inputs as .csv files
save_data = True

# load model and acados_solver
model, acados_solver, acados_integrator = acados_settings(Ts, Tf, N)

# dimensions
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
N_hover = int(T_hover * N / Tf)
N_traj = int(T_traj * N / Tf)
Nsim = int(T * N / Tf)

# initialize data structs
simX = np.ndarray((Nsim+1, nx))
simU = np.ndarray((Nsim, nu))
tot_comp_sum = 0
tcomp_max = 0

# set initial condition for acados integrator
xcurrent = model.x0.reshape((nx,))
simX[0, :] = xcurrent

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('OffboardControl')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Initialize MPC
        # self.N = 20  # number of discretization steps   
        # self.acados_solver = acados_settings(self.N)
        
        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_attitude_setpoint_publisher_ = self.create_publisher(
            VehicleRatesSetpoint,"/fmu/in/vehicle_rates_setpoint", qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback, qos_profile)
        self.hover_thrust_sub = self.create_subscription(
            HoverThrustEstimate, '/fmu/out/hover_thrust_estimate',
            self.hover_thrust_callback, qos_profile)
        self.vehicle_odometry_sub_ = self.create_subscription(
            VehicleOdometry, 'fmu/out/vehicle_odometry', 
            self.vehicle_odometry_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0

        # self.vehicle_odometry = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        # self.hover_thrust_estimate = HoverThrustEstimate()
        
        self.count = 0

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.01, self.timer_callback)

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def hover_thrust_callback(self, hover_thrust_estimate):
        self.hover_thrust_estimate = hover_thrust_estimate
        self.hover_thrust = self.hover_thrust_estimate.hover_thrust

    def vehicle_odometry_callback(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber."""
        self.vehicle_odometry = vehicle_odometry
        self.timestamp = self.vehicle_odometry.timestamp
        self.timestamp_sample = self.vehicle_odometry.timestamp_sample
        self.vehicle_position = ned_to_enu(self.vehicle_odometry.position)
        self.vehicle_velocity = ned_to_enu(self.vehicle_odometry.velocity)
        self.vehicle_q = ned_to_enu_quaternion(self.vehicle_odometry.q)
        self.vehicle_roll, self.vehicle_pitch, self.vehicle_yaw = Quaternion2Euler(self.vehicle_q)
   
    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # (90 degree)
        # msg.yaw = -3.14
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

    def publish_vehicle_attitude_setpoint(self):
        self.count += 1
        print(self.count)
        # euler_z = self.vehicle_yaw
        # if euler_z > np.pi:
        #     euler_z = euler_z - 2*np.pi
        # elif euler_z < -np.pi:
        #     euler_z= euler_z + 2*np.pi
            
        # params = np.array([0.73, 0.155, 0.155, euler_z])

        # set initial condition for acados integrator
        xcurrent = np.array([self.vehicle_position[0], self.vehicle_position[1], self.vehicle_position[2],
                             self.vehicle_q[0], self.vehicle_q[1], self.vehicle_q[2], self.vehicle_q[3],
                             self.vehicle_velocity[0], self.vehicle_velocity[1], self.vehicle_velocity[2]])
        
        trajectory, number_of_steps = read_trajectory_from_file("circle.txt")

        for j in range(N):
            yref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.76, 0.0, 0.0, 0.0])
            acados_solver.set(j, "yref", yref)
            # yref = trajectory[self.count,:11]
            #                 # self.hover_thrust, 1.0, 0.0, 0.0, 0.0])
            # acados_solver.set(j, "yref", yref)
            # acados_solver.set(j, "p", params)
        yref_N = np.array([0.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        acados_solver.set(N, "yref", yref_N)
        # self.acados_solver.set(self.N, "p", params)
        # print("\n\n\n\n\n\n",yref[:6])
        # solve ocp for a fixed reference
        acados_solver.set(0, "lbx", xcurrent)
        acados_solver.set(0, "ubx", xcurrent)

        status = acados_solver.solve()
        if status != 0:
            print("acados returned status {} in closed loop.".format(status))

        # get solution from acados_solver
        xcurrent_pred = acados_solver.get(1, "x")
        u0 = acados_solver.get(0, "u")

        # computed inputs
        Thrust = u0[0]
        Roll = -u0[1]
        Pitch = -u0[2]
        Yaw = -u0[3]

        quaternion = Euler2Quaternion(Roll, Pitch, 0.0)

        Q = np.asfarray(ned_to_enu_quaternion(quaternion), dtype = np.float32)
    
        msg = VehicleRatesSetpoint()

        msg.roll = Roll
        msg.pitch = Pitch
        msg.yaw = Yaw
        msg.thrust_body = np.asfarray([0, 0, -Thrust], dtype = np.float32)  # half-throttle
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)  # time in microseconds
        self.vehicle_attitude_setpoint_publisher_.publish(msg)


        print("x:", u0)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_vehicle_attitude_setpoint()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    def gt_callback(self):
        global real_x, real_y
        global gt_start
        gt_start = True
        real_x.append(self.vehicle_position[0])
        real_y.append(self.vehicle_position[1])
        
        
    # def ref_callback(data):
    #     global ref_start
    #     ref_start = True
    #     global ref_x, ref_y
    #     ref_x.append(data.pose.pose.position.x)
    #     ref_y.append(data.pose.pose.position.y)


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        pass
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)