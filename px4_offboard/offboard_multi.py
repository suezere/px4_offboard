import numpy as np
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import TimesyncStatus, VehicleOdometry, VehicleStatus, HoverThrustEstimate
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleAttitudeSetpoint,TrajectorySetpoint

from acados_settings_model import acados_ocp
from utils import *

# Initialize MPC
N = 40  # number of discretization steps   
acados_solver_1 = acados_ocp(N)
acados_solver_2 = acados_ocp(N)


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

        # Create publishers
        self.offboard_control_mode_publisher_1 = self.create_publisher(
            OffboardControlMode, '/px4_1/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_attitude_setpoint_publisher_1 = self.create_publisher(
            VehicleAttitudeSetpoint,'/px4_1/fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.trajectory_setpoint_publisher_1 = self.create_publisher(
            TrajectorySetpoint, '/px4_1/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher_1 = self.create_publisher(
            VehicleCommand, '/px4_1/fmu/in/vehicle_command', qos_profile)
        self.offboard_control_mode_publisher_2 = self.create_publisher(
            OffboardControlMode, '/px4_2/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_attitude_setpoint_publisher_2 = self.create_publisher(
            VehicleAttitudeSetpoint,"/px4_2/fmu/in/vehicle_attitude_setpoint", qos_profile)
        self.trajectory_setpoint_publisher_2 = self.create_publisher(
            TrajectorySetpoint, '/px4_2/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher_2 = self.create_publisher(
            VehicleCommand, '/px4_2/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_status_sub_1 = self.create_subscription(
            VehicleStatus, '/px4_1/fmu/out/vehicle_status',
            self.vehicle_status_callback_1, qos_profile)
        self.hover_thrust_sub_1 = self.create_subscription(
            HoverThrustEstimate, '/px4_1/fmu/out/hover_thrust_estimate',
            self.hover_thrust_callback_1, qos_profile)
        self.vehicle_odometry_sub_1 = self.create_subscription(
            VehicleOdometry, '/px4_1/fmu/out/vehicle_odometry', 
            self.vehicle_odometry_callback_1, qos_profile)
        self.vehicle_status_sub_2 = self.create_subscription(
            VehicleStatus, '/px4_2/fmu/out/vehicle_status',
            self.vehicle_status_callback_2, qos_profile)
        self.hover_thrust_sub_2 = self.create_subscription(
            HoverThrustEstimate, '/px4_2/fmu/out/hover_thrust_estimate',
            self.hover_thrust_callback_2, qos_profile)
        self.vehicle_odometry_sub_2 = self.create_subscription(
            VehicleOdometry, '/px4_2/fmu/out/vehicle_odometry', 
            self.vehicle_odometry_callback_2, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0

        # self.vehicle_odometry = VehicleOdometry()
        self.vehicle_status_1 = VehicleStatus()
        # self.hover_thrust_estimate = HoverThrustEstimate()
        # self.vehicle_odometry_2 = VehicleOdometry()
        self.vehicle_status_2 = VehicleStatus()
        # self.hover_thrust_estimate_2 = HoverThrustEstimate()

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_status_callback_1(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status_1 = vehicle_status

    def vehicle_status_callback_2(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status_2 = vehicle_status

    def hover_thrust_callback_1(self, hover_thrust_estimate):
        self.hover_thrust_estimate_1 = hover_thrust_estimate
        self.hover_thrust_1 = self.hover_thrust_estimate_1.hover_thrust

    def hover_thrust_callback_2(self, hover_thrust_estimate):
        self.hover_thrust_estimate_2 = hover_thrust_estimate
        self.hover_thrust_2 = self.hover_thrust_estimate_2.hover_thrust

    def vehicle_odometry_callback_1(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber."""
        self.vehicle_odometry_1 = vehicle_odometry
        self.timestamp_1 = self.vehicle_odometry_1.timestamp
        self.timestamp_sample_1 = self.vehicle_odometry_1.timestamp_sample
        self.vehicle_position_1 = ned_to_enu(self.vehicle_odometry_1.position)
        self.vehicle_velocity_1 = ned_to_enu(self.vehicle_odometry_1.velocity)
        self.vehicle_q_1 = ned_to_enu_quaternion(self.vehicle_odometry_1.q)
        self.vehicle_roll_1, self.vehicle_pitch_1, self.vehicle_yaw_1 = Quaternion2Euler(self.vehicle_q_1)

    def vehicle_odometry_callback_2(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber."""
        self.vehicle_odometry_2 = vehicle_odometry
        self.timestamp_2 = self.vehicle_odometry_2.timestamp
        self.timestamp_sample_2 = self.vehicle_odometry_2.timestamp_sample
        self.vehicle_position_2 = ned_to_enu(self.vehicle_odometry_2.position)
        self.vehicle_velocity_2 = ned_to_enu(self.vehicle_odometry_2.velocity)
        self.vehicle_q_2 = ned_to_enu_quaternion(self.vehicle_odometry_2.q)
        self.vehicle_roll_2, self.vehicle_pitch_2, self.vehicle_yaw_2 = Quaternion2Euler(self.vehicle_q_2)

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command_1(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.publish_vehicle_command_2(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0) 
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command_1(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.publish_vehicle_command_2(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command_1(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.publish_vehicle_command_2(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command_1(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.publish_vehicle_command_2(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher_1.publish(msg)
        self.offboard_control_mode_publisher_2.publish(msg)

    def publish_vehicle_attitude_setpoint_1(self):
        euler_z = self.vehicle_yaw_1
        if euler_z > np.pi:
            euler_z = euler_z - 2*np.pi
        elif euler_z < -np.pi:
            euler_z= euler_z + 2*np.pi
            
        params = np.array([0.73, 0.1, 0.1, euler_z])

        # set initial condition for acados integrator
        xcurrent = np.array([self.vehicle_position_1[0], self.vehicle_position_1[1], self.vehicle_position_1[2],
                             self.vehicle_velocity_1[0], self.vehicle_velocity_1[1], self.vehicle_velocity_1[2],
                             self.vehicle_roll_1, self.vehicle_pitch_1])

        for j in range(N):
            yref = np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.73, 0.0, 0.0])
            acados_solver_1.set(j, "yref", yref)
            acados_solver_1.set(j, "p", params)
        yref_N = np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        acados_solver_1.set(N, "yref", yref_N)
        acados_solver_1.set(N, "p", params)

        # solve ocp for a fixed reference
        acados_solver_1.set(0, "lbx", xcurrent)
        acados_solver_1.set(0, "ubx", xcurrent)

        status = acados_solver_1.solve()
        if status != 0:
            print("acados returned status {} in closed loop.".format(status))

        # get solution from acados_solver
        xcurrent_pred = acados_solver_1.get(1, "x")
        u0 = acados_solver_1.get(0, "u")

        # computed inputs
        Thrust = u0[0]
        Roll = u0[1]
        Pitch = u0[2]

        quaternion = Euler2Quaternion(Roll, Pitch, 0.0)

        Q = np.asfarray(ned_to_enu_quaternion(quaternion), dtype = np.float32)
    
        msg = VehicleAttitudeSetpoint()

        msg.q_d = Q  # no rotation
        msg.thrust_body = np.asfarray([0, 0, - Thrust], dtype = np.float32)  # half-throttle
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)  # time in microseconds
        self.vehicle_attitude_setpoint_publisher_1.publish(msg)

        self.get_logger().info(f"x1: {xcurrent[:3]}")
        self.get_logger().info(f"u1: {u0}")

    def publish_vehicle_attitude_setpoint_2(self):
        euler_z = self.vehicle_yaw_2
        if euler_z > np.pi:
            euler_z = euler_z - 2*np.pi
        elif euler_z < -np.pi:
            euler_z= euler_z + 2*np.pi
            
        params = np.array([0.73, 0.1, 0.1, euler_z])

        # set initial condition for acados integrator
        xcurrent = np.array([self.vehicle_position_2[0], self.vehicle_position_2[1], self.vehicle_position_2[2],
                             self.vehicle_velocity_2[0], self.vehicle_velocity_2[1], self.vehicle_velocity_2[2],
                             self.vehicle_roll_2, self.vehicle_pitch_2])
        
        for j in range(N):
            yref = np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.73, 0.0, 0.0])
            acados_solver_2.set(j, "yref", yref)
            acados_solver_2.set(j, "p", params)
        yref_N = np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        acados_solver_2.set(N, "yref", yref_N)
        acados_solver_2.set(N, "p", params)
        
        # solve ocp for a fixed reference
        acados_solver_2.set(0, "lbx", xcurrent)
        acados_solver_2.set(0, "ubx", xcurrent)

        status = acados_solver_2.solve()
        if status != 0:
            print("acados returned status {} in closed loop.".format(status))

        # get solution from acados_solver
        xcurrent_pred = acados_solver_2.get(1, "x")
        u0 = acados_solver_2.get(0, "u")

        # computed inputs
        Thrust = u0[0]
        Roll = u0[1]
        Pitch = u0[2]

        quaternion = Euler2Quaternion(Roll, Pitch, 0.0)

        Q = np.asfarray(ned_to_enu_quaternion(quaternion), dtype = np.float32)
    
        msg = VehicleAttitudeSetpoint()

        msg.q_d = Q  # no rotation
        msg.thrust_body = np.asfarray([0, 0, - Thrust], dtype = np.float32)  # half-throttle
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)  # time in microseconds
        self.vehicle_attitude_setpoint_publisher_2.publish(msg)


        self.get_logger().info(f"x2: {xcurrent[:3]}")
        self.get_logger().info(f"u2: {u0}")

    def publish_vehicle_command_1(self, command, **params) -> None:
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
        msg.target_system = 2
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher_1.publish(msg)

    def publish_vehicle_command_2(self, command, **params) -> None:
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
        msg.target_system = 3
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher_2.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
        self.get_logger().info(f"NAV_STATUS1: {self.vehicle_status_1.nav_state}")
        self.get_logger().info(f"NAV_STATUS2: {self.vehicle_status_2.nav_state}")
        if (self.vehicle_status_1.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
         and self.vehicle_status_2.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
            self.publish_vehicle_attitude_setpoint_1()
            self.publish_vehicle_attitude_setpoint_2()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
