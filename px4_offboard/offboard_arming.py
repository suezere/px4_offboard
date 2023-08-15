"""
Python implementation of Offboard Control

"""
import numpy as np
import rclpy 
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy import utilities
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import TimesyncStatus, VehicleOdometry, VehicleStatus, HoverThrustEstimate
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleAttitudeSetpoint,TrajectorySetpoint

from acados_settings_model import acados_ocp
from utils import *

# Initialize MPC
N = 40  # number of discretization steps   
acados_solver = acados_ocp(N)

class OffboardControl(Node):

    def __init__(self) -> None:
        super().__init__('OffboardControl')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        # ROS2 SUBSCRIBERS 
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback, qos_profile)
        self.hover_thrust_sub = self.create_subscription(
            HoverThrustEstimate, '/fmu/out/hover_thrust_estimate',
            self.hover_thrust_callback, qos_profile)
        self.vehicle_odometry_sub_ = self.create_subscription(
            VehicleOdometry, 'fmu/out/vehicle_odometry', 
            self.vehicle_odometry_callback, qos_profile)

        # ROS2 PUBLISHERS 
        self.offboard_control_mode_publisher_ = self.create_publisher(OffboardControlMode,"/fmu/in/offboard_control_mode", qos_profile)
        self.vehicle_attitude_setpoint_publisher_ = self.create_publisher(VehicleAttitudeSetpoint,"/fmu/in/vehicle_attitude_setpoint", qos_profile)
        self.vehicle_command_publisher_ = self.create_publisher(VehicleCommand,"/fmu/in/vehicle_command", qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        
        self.mission_state = 0
        self.offboard_setpoint_counter = 0

        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        self.hover_thrust_estimate = HoverThrustEstimate()

        self.takeoff_height = -5.0

        # Timers
        # self.timer_offboard = self.create_timer(0.5, self.timer_offboard_cb)
        # self.timer_mission = self.create_timer(10, self.mission_cb)
        self.timer = self.create_timer(0.1, self.timer_cb)

    # def vehicle_status_callback(self, msg):
    #     self.get_logger().info(f"NAV_STATUS: {msg.nav_state}")
    #     self.get_logger().info(f"ARM STATUS: {msg.arming_state}")
    #     # self.get_logger().info(f"FlightCheck: {msg.pre_flight_checks_pass}")

    #     self.nav_state = msg.nav_state
    #     self.arm_state = msg.arming_state
    #     self.failsafe = msg.failsafe
    #     self.flightCheck = msg.pre_flight_checks_pass
    def vehicle_status_callback(self, vehicle_status):
        self.get_logger().info(f"NAV_STATUS: {vehicle_status.nav_state}")
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def hover_thrust_callback(self, hover_thrust_estimate):
        self.hover_thrust = hover_thrust_estimate

    def vehicle_odometry_callback(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber."""
        self.vehicle_odometry = vehicle_odometry
        self.timestamp = self.vehicle_odometry.timestamp
        self.timestamp_sample = self.vehicle_odometry.timestamp_sample
        self.vehicle_x = self.vehicle_odometry.position[0]
        self.vehicle_y = self.vehicle_odometry.position[1]      
        self.vehicle_z = - self.vehicle_odometry.position[2]
        self.vehicle_u = self.vehicle_odometry.velocity[0]
        self.vehicle_v = self.vehicle_odometry.velocity[1]  
        self.vehicle_w = - self.vehicle_odometry.velocity[2]
        self.vehicle_roll, self.vehicle_pitch, self.vehicle_yaw = Quaternion2Euler(self.vehicle_odometry.q)
        self.vehicle_yaw = np.pi/2 - self.vehicle_yaw
        # self.vehicle_wroll = self.vehicle_odometry.angular_velocity[0]
        # self.vehicle_wpitch = self.vehicle_odometry.angular_velocity[1]
        # self.vehicle_wyaw = self.vehicle_odometry.angular_velocity[2]
     
    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Arm command sent")

    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info("Disarm command sent")

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")
    
    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_mode(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.offboard_control_mode_publisher_.publish(msg)

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

        ref_traj = np.array([1,1,-1])
        params = np.array([self.hover_thrust.hover_thrust, 0.1, 0.1, 0])

        # set initial condition for acados integrator
        xcurrent = np.array([self.vehicle_x, self.vehicle_y, self.vehicle_z,
                             self.vehicle_u,self.vehicle_v,self.vehicle_w,
                             self.vehicle_roll,self.vehicle_pitch])
        print("\\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nn\n\n\n\n\n\n\n\n\n\n\n\nkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk", self.hover_thrust.hover_thrust)


        for j in range(N):
            yref = np.array([1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             self.hover_thrust.hover_thrust, 0.0, 0.0])
                            # self.hover_thrust, 1.0, 0.0, 0.0, 0.0])
            acados_solver.set(j, "yref", yref)
            acados_solver.set(j, "p", params)
        yref_N = np.array([1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        acados_solver.set(N, "yref", yref_N)
        acados_solver.set(N, "p", params)
        
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
        Roll = u0[1]
        Pitch = u0[2]

        quaternion = Euler2Quaternion(Roll, Pitch, np.pi/2)
        print(Thrust)
        # making sure that q is normalized
        # quaternion = unit_quat(quaternion)

        msg = VehicleAttitudeSetpoint()

        msg.q_d = quaternion  # no rotation
        msg.thrust_body = np.asfarray([0, 0, Thrust], dtype = np.float32)  # half-throttle
        msg.timestamp = int(Clock().now().nanoseconds / 1000)  # time in microseconds
        self.vehicle_attitude_setpoint_publisher_.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:       
        msg = VehicleCommand()
        msg.command = command  # command ID
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1  # system which should execute the command
        msg.target_component = 1  # component which should execute the command, 0 for all components
        msg.source_system = 1  # system sending the command
        msg.source_component = 1  # component sending the command
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.vehicle_command_publisher_.publish(msg)
    
    def mission_cb(self):
        """ Funzione mission organizzata come macchina a stati """

        if self.mission_state == 0:
            """ Arming and no moving setpoint """
            self.get_logger().info("Mission started")
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.mission_state = 1

        elif self.mission_state == 1:    
            self.arm()
            self.get_logger().info("Vehicle armed")
            self.mission_state = 2
        
        elif self.mission_state == 2:
            self.get_logger().info("Attitude Setpoint sent")
            self.publish_vehicle_attitude_setpoint()
            self.mission_state = 3 

        # elif self.mission_state == 3:
        #     """Landing"""
        #     self.get_logger().info("Landing request")
        #     self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        #     self.mission_state = 4
        
        # elif self.mission_state == 4:
        #     self.timer_offboard.cancel()
        #     self.get_logger().info("Mission finished")
        #     self.timer_mission.cancel()
        #     exit()  
    
    def timer_offboard_cb(self):
        self.publish_offboard_control_mode()

    def timer_cb(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_mode()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        # self.publish_vehicle_attitude_setpoint()


        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            # self.publish_vehicle_attitude_setpoint()
            self.publish_position_setpoint(1.0, 1.0, -3.0)

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
   

def main(args=None) -> None:
    
    rclpy.init(args=args)
    print("Starting offboard control node...\n")

    
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
