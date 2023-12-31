import numpy as np
import time
import mavros
import message_filters
import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import AttitudeTarget
from px4_msgs.msg import TimesyncStatus, VehicleOdometry, VehicleStatus, HoverThrustEstimate
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleLocalPosition, TrajectorySetpoint

from utils import *


class SystemIdentification(Node):
    """Node for a vehicle in system identification."""

    def __init__(self) -> None:
        super().__init__('system_identification')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
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
        self.attitude_sub = message_filters.Subscriber(self,PoseStamped,'/mavros/local_position/pose')
        self.attitude_target_sub = message_filters.Subscriber(self,AttitudeTarget,'/mavros/setpoint_raw/target_attitude')
        self.attitude_angular_rate_sub = message_filters.Subscriber(self,TwistStamped,'/mavros/local_position/velocity_local')
        self.time_synchronizer = message_filters.TimeSynchronizer([self.attitude_sub, self.attitude_target_sub,self.attitude_angular_rate_sub],qos_profile)
        self.time_synchronizer.registerCallback(self.vehicle_sync_callback)

        # Initialize variables
        self.takeoff_height = -3.0
        self.yaw_ref = np.pi / 4
        self.tau_roll_id = False
        self.tau_pitch_id = False
        self.tau_yaw_id = False
        self.tau_roll_diff = []
        self.tau_roll_rate = []
        self.tau_pitch_diff = []
        self.tau_pitch_rate = []
        self.tau_yaw_diff = []
        self.tau_yaw_rate = []
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()

        #Define constants
        self.TAKEOFF = 0
        self.HOVER = 1
        self.X_MANEUVER = 2
        self.LAND = 3
        self.FINISH = 4
        self.state = self.TAKEOFF

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.02, self.timer_callback)

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

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def vehicle_sync_callback(self, attitude, attitude_target, attitude_angular_rate):
        target_x, target_y,target_z = Quaternion2Euler(attitude_target.pose.orientation)
        current_x, current_y, current_z = Quaternion2Euler(attitude.orientation)
        if self.tau_roll_id:
            self.tau_roll_diff.append(target_x - current_x)
            self.tau_roll_rate.append(attitude_angular_rate.angular_velocity[0])
        elif self.tau_pitch_id:
            self.tau_pitch_diff.append(target_y - current_y)
            self.tau_pitch_rate.append(attitude_angular_rate.angular_velocity[1])
        elif self.tau_yaw_id:
            self.tau_yaw_diff.append(target_z - current_z)
            self.tau_yaw_rate.append(attitude_angular_rate.angular_velocity[2])
        
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
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
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
            self.takeoff_pose_x = self.vehicle_position[0]
            self.takeoff_pose_y = self.vehicle_position[1]
            self.takeoff_pose_z = self.vehicle_position[2] + self.takeoff_height
            
            if self.state == self.TAKEOFF:
                self.publish_position_setpoint(self.takeoff_pose_x, self.takeoff_pose_y, self.takeoff_pose_z)
                print(self.vehicle_position[2])
                # if self.target_reached(self.takeoff_pose_x, self.takeoff_pose_y, self.takeoff_pose_z):
                #     self.land
                
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

        self.last_state_time = self.get_clock().now().nanoseconds

    def target_reached(self, x, y, z):
        return np.sqrt((self.vehicle_position[0] - x)**2 + (self.vehicle_position[1] - y)**2 + (self.vehicle_position[2] - z)**2) <2

    def update_x_maneuver(self):
        self.x_maneuver_pose_x = self.vehicle_position[0] + 1.5*np.sin(2*(self.get_clock().now().nanoseconds * 1e-9 - self.last_state_time * 1e-9))
        self.x_maneuver_pose_y = self.vehicle_position[1]
        self.x_maneuver_pose_z = self.vehicle_position[2]
        self.x_maneuver_q_w = 1
        self.x_maneuver_q_x = 0
        self.x_maneuver_q_y = 0
        self.x_maneuver_q_z = 0

    def update_y_maneuver(self):
        self.y_maneuver_pose_x = self.vehicle_position[0]
        self.y_maneuver_pose_y = self.vehicle_position[1] + 1.5*np.sin(2*(self.get_clock().now().nanoseconds * 1e-9 - self.last_state_time * 1e-9))
        self.y_maneuver_pose_z = self.vehicle_position[2]
        self.y_maneuver_q_w = 1
        self.y_maneuver_q_x = 0
        self.y_maneuver_q_y = 0
        self.y_maneuver_q_z = 0

    def update_yaw_maneuver(self):
        euler_x, euler_y,euler_z = Quaternion2Euler(self.vehicle_q)
        # Eigen::Vector3d euler = q2rpy(local_pose.pose.orientation)
        if np.pi/4 - np.pi/16 < euler_z < np.pi/4 + np.pi/16 and self.yaw_ref == np.pi/4:
            self.yaw_ref = -np.pi/4
        elif -np.pi/4 - np.pi/16 < euler_z < -np.pi/4 + np.pi/16 and self.yaw_ref == -np.pi/4:
            self.yaw_ref = np.pi/4
        
        self.yaw_maneuver_q = Euler2Quaternion(0, 0, self.yaw_ref)
        self.yaw_maneuver_pose_x = self.vehicle_position[0]
        self.yaw_maneuver_pose_x = self.vehicle_position[1]
        self.yaw_maneuver_pose_x = self.vehicle_position[2]

    def linear(x,y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        numerator = 0
        denominator = 0

        for i in range(len(x)):
            numerator += (x[i] - mean_x) * (y[i] - mean_y)
            denominator += (x[i] - mean_x) * (x[i] - mean_x)

        a = numerator / denominator
        return a
    




def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    system_identification = SystemIdentification()
    rclpy.spin(system_identification)
    system_identification.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
