#! /usr/bin/env python
import os
from collections import deque
import numpy as np
import time

from brush_stroke import euler_from_quaternion

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

class Robot:
    '''
        Low-level action functionality of the robot.
        This is an abstract class, see its children for usage.
    '''
    def __init__(self, debug, node_name="painting"):
        self.debug_bool = debug
        import rospy
        rospy.init_node(node_name)

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        raise Exception("This method must be implemented")

    def good_night_robot(self):
        raise Exception("This method must be implemented")

    def go_to_cartesian_pose(self, positions, orientations, precise=False):
        raise Exception("This method must be implemented")

class XArm(Robot, object):
    '''
        Low-level action functionality of the robot.
        This is an abstract class, see its children for usage.
    '''
    def __init__(self, ip, debug):
        from xarm.wrapper import XArmAPI
        self.debug_bool = debug

        self.arm = XArmAPI(ip)

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        self.arm.motion_enable(enable=True)
        self.arm.reset(wait=True)
        self.arm.set_mode(0)
        self.arm.reset(wait=True)
        self.arm.set_state(state=0)

        self.arm.reset(wait=True)

    def good_night_robot(self):
        self.arm.disconnect()

    def go_to_cartesian_pose(self, positions, orientations,
            speed=250):
        # positions in meters
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None,:]
            orientations = orientations[None,:]
        for i in range(len(positions)):
            x,y,z = positions[i][1], positions[i][0], positions[i][2]
            x,y,z = x*1000, y*-1000, z*1000 #m to mm
            q = orientations[i]
            
            euler= euler_from_quaternion(q[0], q[1], q[2], q[3])#quaternion.as_quat_array(orientations[i])
            roll, pitch, yaw = 180, 0, 0#euler[0], euler[1], euler[2]
            # https://github.com/xArm-Developer/xArm-Python-SDK/blob/0fd107977ee9e66b6841ea9108583398a01f227b/xarm/x3/xarm.py#L214
            
            wait = True 
            failure, state = self.arm.get_position()
            if not failure:
                curr_x, curr_y, curr_z = state[0], state[1], state[2]
                # print('curr', curr_y, y)
                dist = ((x-curr_x)**2 + (y-curr_y)**2 + (z-curr_z)**2)**0.5
                # print('dist', dist)
                # Dist in mm
                if dist < 5:
                    wait=False
                    speed=600
                    # print('less')

            try:
                r = self.arm.set_position(
                        x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                        speed=speed, wait=wait
                )
                # print(r)
                if r:
                    print("failed to go to pose, resetting.")
                    self.arm.clean_error()
                    self.good_morning_robot()
                    self.arm.set_position(
                            x=x, y=y, z=z+5, roll=roll, pitch=pitch, yaw=yaw,
                            speed=speed, wait=True
                    )
                    self.arm.set_position(
                            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                            speed=speed, wait=True
                    )
            except Exception as e:
                self.good_morning_robot()
                print('Cannot go to position', e)


class Franka(Robot, object):
    '''
        Low-level action functionality of the Franka robot.
    '''
    def __init__(self, debug, node_name="painting"):
        import sys
        sys.path.append('~/Documents/frankapy/frankapy/')
        from frankapy import FrankaArm

        self.debug_bool = debug
        self.fa = FrankaArm()

        # reset franka to its home joints
        self.fa.reset_joints()

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        # reset franka to its home joints
        self.fa.reset_joints()

    def good_night_robot(self):
        # reset franka back to home
        self.fa.reset_joints()

    def create_rotation_transform(pos, quat):
        from autolab_core import RigidTransform
        rot = RigidTransform.rotation_from_quaternion(quat)
        rt = RigidTransform(rotation=rot, translation=pos,
                from_frame='franka_tool', to_frame='world')
        return rt

    def sawyer_to_franka_position(pos):
        # Convert from sawyer code representation of X,Y to Franka
        pos[0] *= -1 # The x is oposite sign from the sawyer code
        pos[:2] = pos[:2][::-1] # The x and y are switched compared to sawyer for which code was written
        return pos
    
    def go_to_cartesian_pose(self, positions, orientations):
        """
            Move to a list of points in space
            args:
                positions (np.array(n,3)) : x,y,z coordinates in meters from robot origin
                orientations (np.array(n,4)) : x,y,z,w quaternion orientation
                precise (bool) : use precise for slow short movements. else use False, which is fast but unstable
        """
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None,:]
            orientations = orientations[None,:]

        # if precise:
        #     self.go_to_cartesian_pose_precise(positions, orientations, stiffness_factor=stiffness_factor)
        # else:
        #     self.go_to_cartesian_pose_stable(positions, orientations)
        self.go_to_cartesian_pose_stable(positions, orientations)


    def go_to_cartesian_pose_stable(self, positions, orientations):
        abs_diffs = []
        # start = time.time()
        for i in range(len(positions)):
            pos = Franka.sawyer_to_franka_position(positions[i])
            rt = Franka.create_rotation_transform(pos, orientations[i])

            # Determine speed/duration
            curr_pos = self.fa.get_pose().translation
            # print(curr_pos, pos)
            dist = ((curr_pos - pos)**2).sum()**.5 
            # print('distance', dist*100)
            duration = dist * 5 # 1cm=.1s 1m=10s
            duration = max(0.6, duration) # Don't go toooo fast
            if dist*100 < 0.8: # less than 0.8cm
                # print('very short')
                duration = 0.3
            # print('duration', duration, type(duration))
            duration = float(duration)
            if pos[2] < 0.05:
                print('below threshold!!', pos[2])
                continue
            try:
                self.fa.goto_pose(rt,
                        duration=duration, 
                        force_thresholds=[10,10,10,10,10,10],
                        ignore_virtual_walls=True,
                        buffer_time=0.0
                )    
            except Exception as e:
                print('Could not goto_pose', e)
            abs_diff = sum((self.fa.get_pose().translation-rt.translation)**2)**0.5 * 100
            # print(abs_diff, 'cm stable')
            abs_diffs.append(abs_diff)
        # print(max(abs_diffs), sum(abs_diffs)/len(abs_diffs), '\t', time.time() - start)
        if abs_diffs[-1] > 1:
            # Didn't get to the last position. Try again.
            print('Off from final position by', abs_diffs[-1], 'cm')
            self.fa.goto_pose(rt,
                        duration=duration+3, 
                        force_thresholds=[10,10,10,10,10,10],
                        ignore_virtual_walls=True,
                        buffer_time=0.0
                )
            abs_diff = sum((self.fa.get_pose().translation-rt.translation)**2)**0.5 * 100
            if abs_diff > 1:
                print('Failed to get to end of trajectory again. Resetting Joints')
                self.fa.reset_joints()

    def go_to_cartesian_pose_precise(self, positions, orientations, hertz=300, stiffness_factor=3.0):
        """
            This is a smooth version of this function. It can very smoothly go betwen the positions.
            However, it is unstable, and will result in oscilations sometimes.
            Recommended to be used only for fine, slow motions like the actual brush strokes.
        """
        start = time.time()
        from frankapy import SensorDataMessageType
        from frankapy import FrankaConstants as FC
        from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
        from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage
        from franka_interface_msgs.msg import SensorDataGroup
        from frankapy.utils import min_jerk, min_jerk_weight
        import rospy
        
        def get_duration(here, there):
            dist = ((here.translation - there.translation)**2).sum()**.5
            duration = dist *  10#5 # 1cm=.1s 1m=10s
            duration = max(0.4, duration) # Don't go toooo fast
            duration = float(duration)
            return duration, dist

        def smooth_trajectory(poses, window_width=50):
            # x = np.cumsum(delta_ts)
            from scipy.interpolate import interp1d
            from scipy.ndimage import gaussian_filter1d

            for c in range(3):
                coords = np.array([p.translation[c] for p in poses])

                coords_smooth = gaussian_filter1d(coords, 31)
                print(len(poses), len(coords_smooth))
                for i in range(len(poses)-1):
                    coords_smooth[i]
                    poses[i].translation[c] = coords_smooth[i]
            return poses

        pose_trajs = []
        delta_ts = []
        
        # Loop through each position/orientation and create interpolations between the points
        p0 = self.fa.get_pose()
        for i in range(len(positions)):
            p1 = Franka.create_rotation_transform(\
                Franka.sawyer_to_franka_position(positions[i]), orientations[i])

            duration, distance = get_duration(p0, p1)

            # needs to be high to avoid torque discontinuity error controller_torque_discontinuity
            STEPS = max(10, int(duration*hertz))
            # print(STEPS, distance)

            # if distance*100 > 5:
            #     print("You're using the precise movement wrong", distance*100)

            ts = np.arange(0, duration, duration/STEPS)
            # ts = np.linspace(0, duration, STEPS)
            weights = [min_jerk_weight(t, duration) for t in ts]

            if i == 0 or i == len(positions)-1:
                # Smooth for the first and last way points
                pose_traj = [p0.interpolate_with(p1, w) for w in weights]
            else:
                # linear for middle points cuz it's fast and accurate
                pose_traj = p0.linear_trajectory_to(p1, STEPS)
            # pose_traj = [p0.interpolate_with(p1, w) for w in weights]
            # pose_traj = p0.linear_trajectory_to(p1, STEPS)

            pose_trajs += pose_traj
            
            delta_ts += [duration/len(pose_traj),]*len(pose_traj)
            
            p0 = p1
            
        T = float(np.array(delta_ts).sum())

        # pose_trajs = smooth_trajectory(pose_trajs)

        pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        
        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        self.fa.goto_pose(pose_trajs[1], duration=T, dynamic=True, 
            buffer_time=T+10,
            force_thresholds=[10,10,10,10,10,10],
            cartesian_impedances=(np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES)*stiffness_factor).tolist() + FC.DEFAULT_ROTATIONAL_STIFFNESSES,
            ignore_virtual_walls=True,
        )
        abs_diffs = []
        try:
            init_time = rospy.Time.now().to_time()
            for i in range(2, len(pose_trajs)):
                timestamp = rospy.Time.now().to_time() - init_time
                traj_gen_proto_msg = PosePositionSensorMessage(
                    id=i, timestamp=timestamp, 
                    position=pose_trajs[i].translation, quaternion=pose_trajs[i].quaternion
                )
                fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                    id=i, timestamp=timestamp,
                    # translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:2] + [z_stiffness_trajs[i]],
                    translational_stiffnesses=(np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES)*stiffness_factor).tolist(),
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
                )
                
                ros_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                    feedback_controller_sensor_msg=sensor_proto2ros_msg(
                        fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )

                # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
                pub.publish(ros_msg)
                # rate = rospy.Rate(1 / (delta_ts[i]))
                rate = rospy.Rate(hertz)
                rate.sleep()

                # if i%100==0:
                #     print(self.fa.get_pose().translation[-1] - pose_trajs[i].translation[-1], 
                #         '\t', self.fa.get_pose().translation[-1], '\t', pose_trajs[i].translation[-1])
                if i%10 == 0:
                    abs_diff = sum((self.fa.get_pose().translation-pose_trajs[i].translation)**2)**0.5 * 100
                    # print(abs_diff, 'cm')
                    abs_diffs.append(abs_diff)
                    # print(self.fa.get_pose().translation[-1] - pose_trajs[i].translation[-1], 
                    #     '\t', self.fa.get_pose().translation[-1], '\t', pose_trajs[i].translation[-1])
        except Exception as e:
            print('unable to execute skill', e)
        print(max(abs_diffs), sum(abs_diffs)/len(abs_diffs), '\t', time.time() - start)
        # Stop the skill
        self.fa.stop_skill()
        
class SimulatedRobot(Robot, object):
    def __init__(self, debug=True):
        pass

    def good_morning_robot(self):
        pass

    def good_night_robot(self):
        pass

    def go_to_cartesian_pose(self, position, orientation):
        pass



class Sawyer(Robot, object):
    def __init__(self, debug=True):
        super(Sawyer, self).__init__(debug)
        import rospy


        from intera_core_msgs.srv import (
            SolvePositionIK,
            SolvePositionIKRequest,
            SolvePositionFK,
            SolvePositionFKRequest,
        )
        import intera_interface
        from intera_interface import CHECK_VERSION
        import PyKDL
        from tf_conversions import posemath

        self.limb = intera_interface.Limb(synchronous_pub=False)
        # print(self.limb)

        self.ns = "ExternalTools/right/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(self.ns, SolvePositionIK, persistent=True)
        rospy.wait_for_service(self.ns, 5.0)

    def good_morning_robot(self):
        import intera_interface
        import rospy
        self.debug("Getting robot state... ")
        rs = intera_interface.RobotEnable(False)
        init_state = rs.state().enabled
        self.debug("Enabling robot... ")
        rs.enable()

        def clean_shutdown():
            """
            Exits example cleanly by moving head to neutral position and
            maintaining start state
            """
            self.debug("\nExiting example...")
            limb = intera_interface.Limb(synchronous_pub=True)
            limb.move_to_neutral(speed=.2)
            # 1/0

        rospy.on_shutdown(clean_shutdown)
        self.debug("Excecuting... ")

        # neutral_pose = rospy.get_param("named_poses/{0}/poses/neutral".format(self.name))
        # angles = dict(list(zip(self.joint_names(), neutral_pose)))
        # self.set_joint_position_speed(0.1)
        # self.move_to_joint_positions(angles, timeout)
        intera_interface.Limb(synchronous_pub=True).move_to_neutral(speed=.2)
        
        return rs

    def good_night_robot(self):
        import rospy
        """ Tuck it in, read it a story """
        rospy.signal_shutdown("Example finished.")
        self.debug("Done")

    def go_to_cartesian_pose(self, position, orientation):
        #if len(position)
        position, orientation = np.array(position), np.array(orientation)
        if len(position.shape) == 1:
            position = position[None,:]
            orientation = orientation[None,:]

        # import rospy
        # import argparse
        from intera_motion_interface import (
            MotionTrajectory,
            MotionWaypoint,
            MotionWaypointOptions
        )
        from intera_motion_msgs.msg import TrajectoryOptions
        from geometry_msgs.msg import PoseStamped
        import PyKDL
        # from tf_conversions import posemath
        # from intera_interface import Limb

        limb = self.limb#Limb()

        traj_options = TrajectoryOptions()
        traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
        traj = MotionTrajectory(trajectory_options = traj_options, limb = limb)

        wpt_opts = MotionWaypointOptions(max_linear_speed=0.8*1.5,
                                         max_linear_accel=0.8*1.5,
                                         # joint_tolerances=0.05,
                                         corner_distance=0.005,
                                         max_rotational_speed=1.57,
                                         max_rotational_accel=1.57,
                                         max_joint_speed_ratio=1.0)

        for i in range(len(position)):
            waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)

            joint_names = limb.joint_names()

            endpoint_state = limb.tip_state("right_hand")
            pose = endpoint_state.pose

            pose.position.x = position[i,0]
            pose.position.y = position[i,1]
            pose.position.z = position[i,2]

            pose.orientation.x = orientation[i,0]
            pose.orientation.y = orientation[i,1]
            pose.orientation.z = orientation[i,2]
            pose.orientation.w = orientation[i,3]

            poseStamped = PoseStamped()
            poseStamped.pose = pose

            joint_angles = limb.joint_ordered_angles()
            waypoint.set_cartesian_pose(poseStamped, "right_hand", joint_angles)

            traj.append_waypoint(waypoint.to_msg())

        result = traj.send_trajectory(timeout=None)
        # print(result.result)
        success = result.result
        # if success != True:
        #     print(success)
        if not success:
            import time
            # print('sleeping')
            time.sleep(2)
            # print('done sleeping. now to neutral')
            # Go to neutral and try again
            limb.move_to_neutral(speed=.3)
            # print('done to neutral')
            result = traj.send_trajectory(timeout=None)
            # print('just tried to resend trajectory')
            if result.result:
                print('second attempt successful')
            else:
                print('failed second attempt')
            success = result.result
        return success

    def move_to_joint_positions(self, position, timeout=3, speed=0.1):
        """
        args:
            dict{'right_j0',float} - dictionary of joint to joint angle
        """
        # rate = rospy.Rate(100)
        #try:
        # print('Positions:', position)
        self.limb.set_joint_position_speed(speed=speed)
        self.limb.move_to_joint_positions(position, timeout=timeout,
                                     threshold=0.008726646)
        self.limb.set_joint_position_speed(speed=.1)
        # rate.sleep()
        # except Exception as e:
        #     print('Exception while moving robot:\n', e)
        #     import traceback
        #     import sys
        #     print(traceback.format_exc())


    def display_image(self, file_path):
        import intera_interface
        head_display = intera_interface.HeadDisplay()
        # display_image params:
        # 1. file Path to image file to send. Multiple files are separated by a space, eg.: a.png b.png
        # 2. loop Display images in loop, add argument will display images in loop
        # 3. rate Image display frequency for multiple and looped images.
        head_display.display_image(file_path, False, 100)
    def display_frida(self):
        import rospkg
        rospack = rospkg.RosPack()
        # get the file path for rospy_tutorials
        ros_dir = rospack.get_path('paint')
        self.display_image(os.path.join(str(ros_dir), 'src', 'frida.jpg'))

    def take_picture(self):
        import cv2
        from cv_bridge import CvBridge, CvBridgeError
        import matplotlib.pyplot as plt
        def show_image_callback(img_data):
            """The callback function to show image by using CvBridge and cv
            """
            bridge = CvBridge()
            try:
                cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
            except CvBridgeError as err:
                rospy.logerr(err)
                return

            # edge_str = ''
            # cv_win_name = ' '.join(['heyyyy', edge_str])
            # cv2.namedWindow(cv_win_name, 0)
            # refresh the image on the screen
            # cv2.imshow(cv_win_name, cv_image)
            # cv2.waitKey(3)
            plt.imshow(cv_image[:,:,::-1])
            plt.show()
        rp = intera_interface.RobotParams()
        valid_cameras = rp.get_camera_names()
        print('valid_cameras', valid_cameras)

        camera = 'head_camera'
        # camera = 'right_hand_camera'
        cameras = intera_interface.Cameras()
        if not cameras.verify_camera_exists(camera):
            rospy.logerr("Could not detect the specified camera, exiting the example.")
            return
        rospy.loginfo("Opening camera '{0}'...".format(camera))
        cameras.start_streaming(camera)
        cameras.set_callback(camera, show_image_callback,
            rectify_image=False)
        raw_input('Attach the paint brush now. Press enter to continue:')
        
"""Imports for SO-101"""
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from motion_planner import SO101MotionPlanner

class SO101(Robot, object):
    '''
        Low-level action functionality of the SO-101 robot.
    '''
    def __init__(self, debug, node_name="painting", show_viewer=False):
        self.debug_bool = debug
        
        # Config for arm
        config = SO101FollowerConfig(
            port="/dev/tty.usbmodem5B8E1155331",
            id="my_arm",
            use_degrees=True,
            # cameras={"top": 0, "side": 1},
            max_relative_target=None, # Limit max motion to 10 degrees
        )
        self.arm = SO101Follower(config=config)
        
        self.motor_names = list(self.arm.bus.motors.keys())
        self.arm_joint_names = [n for n in self.motor_names if n != "gripper"]
        self._last_arm_command_deg = None
        self.motion_planner = SO101MotionPlanner(
            show_viewer=show_viewer, lerobot_calibration_path=self.arm.calibration_fpath,
        )

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        self._last_arm_command_deg = None
        self.arm.connect()
        
        with self.arm.bus.torque_disabled():
            self.arm.bus.write("P_Coefficient", "shoulder_lift", 88)
            self.arm.bus.write("P_Coefficient", "elbow_flex", 56)
            self.arm.bus.write("P_Coefficient", "shoulder_pan", 48)
            self.arm.bus.write("P_Coefficient", "wrist_flex", 56)
            self.arm.bus.write("P_Coefficient", "wrist_roll", 48)

        # Session settings, applied after LeRobot's connection/configuration.
        # Keep EEPROM locked, matching the successful dead-zone experiments.
        for name in self.arm_joint_names:
            for register in ("CW_Dead_Zone", "CCW_Dead_Zone"):
                self.arm.bus.write(register, name, 0, normalize=False)
                if self.arm.bus.read(register, name, normalize=False) != 0:
                    raise RuntimeError(f"Failed to set {name} {register} to zero")
            
    def good_night_robot(self):
        self._last_arm_command_deg = None
        self.arm.disconnect()
        
    @staticmethod
    def position_equal(target, actual, tolerance=1.0) -> bool:
        return all(
            abs(float(actual[k]) - float(v)) <= tolerance
            for k, v in target.items()
            if k.endswith(".pos")
        )
    
    def frida_to_robot(self, positions: np.ndarray) -> np.ndarray:
        # Transformation matrix mapping [x, y, z] -> [y, -x, z]
        transform = np.array([
            [ 0, -1,  0],  # New X = Old Y
            [ 1,  0,  0],  # New Y = -Old X
            [ 0,  0,  1],  # New Z = Old Z
        ])
        return positions @ transform
    
    def execute_trajectory(
        self, times, joint_traj, tolerance_deg=2.0, timeout=3.0
    ):
        # An interrupted/failed move must never leave a reusable stale endpoint.
        self._last_arm_command_deg = None
        if len(times) != len(joint_traj):
            raise ValueError("Each timestamp must have one joint command")
        if not joint_traj:
            return

        start_time = time.perf_counter()

        for target_time, joint_action in zip(times, joint_traj):
            delay = start_time + float(target_time) - time.perf_counter()

            if delay > 0:
                time.sleep(delay)

            self.arm.send_action(joint_action)

        # Wait for the arm to reach the final target.
        final_target = joint_traj[-1]
        settle_start = time.perf_counter()
        deadline = settle_start + timeout
        next_report = settle_start
        recent_positions = deque()

        while True:
            observation = self.arm.get_observation()
            now = time.perf_counter()
            actual = {
                name: float(observation[f"{name}.pos"])
                for name in self.arm_joint_names
            }
            signed_errors = {
                name: float(final_target[f"{name}.pos"]) - actual[name]
                for name in self.arm_joint_names
            }
            errors = {name: abs(error) for name, error in signed_errors.items()}
            # Retain roughly the last half-second, including the sample just
            # before its boundary. Range captures oscillation as well as drift.
            recent_positions.append((now, actual))
            while len(recent_positions) > 2 and recent_positions[1][0] < now - 0.5:
                recent_positions.popleft()

            if max(errors.values()) <= tolerance_deg:
                self._last_arm_command_deg = {
                    name: float(final_target[f"{name}.pos"])
                    for name in self.arm_joint_names
                }
                return

            failing = [name for name, error in errors.items() if error > tolerance_deg]
            if now >= next_report:
                details = "; ".join(
                    f"{name}: target={float(final_target[f'{name}.pos']):+.2f}°, "
                    f"actual={actual[name]:+.2f}°, error={signed_errors[name]:+.2f}°"
                    for name in failing
                )
                print(f"[SO101 settling {now - settle_start:.2f}/{timeout:.2f}s] {details}")
                next_report = now + 0.5

            if now >= deadline:
                window = now - recent_positions[0][0]
                details = [
                    f"Final target not reached after {now - settle_start:.2f}s "
                    f"of settling (tolerance={tolerance_deg:.2f}°).",
                    "Angles below are LeRobot servo degrees; signed error = target - actual.",
                ]
                for name in self.arm_joint_names:
                    values = [positions[name] for _, positions in recent_positions]
                    details.append(
                        f"  {name}: target={float(final_target[f'{name}.pos']):+.3f}°, "
                        f"actual={actual[name]:+.3f}°, error={signed_errors[name]:+.3f}°, "
                        f"recent_range={max(values) - min(values):.3f}° "
                        f"over {window:.2f}s"
                        + (" -- OUTSIDE tolerance" if name in failing else "")
                    )
                # Best-effort reads only, after the timeout. They must neither
                # alter torque/gains nor hide the original tracking failure.
                details.extend(self._read_timeout_motor_diagnostics(failing))
                raise TimeoutError("\n".join(details))

            time.sleep(0.02)

    def _read_timeout_motor_diagnostics(self, joint_names):
        """Read the failed joints' registers without writing or reconfiguring."""
        registers = [
            ("Goal_Position_deg", "Goal_Position", True),
            ("Goal_Position_raw", "Goal_Position", False),
        ] + [
            (name, name, False) for name in (
                "Min_Position_Limit", "Max_Position_Limit", "Operating_Mode",
                "P_Coefficient", "I_Coefficient", "D_Coefficient",
                "CW_Dead_Zone", "CCW_Dead_Zone",
                "Torque_Enable", "Torque_Limit", "Max_Torque_Limit",
                "Present_Velocity", "Present_Load", "Present_Current",
                "Present_Voltage", "Present_Temperature", "Status", "Moving",
            )
        ]
        readings = {name: [] for name in joint_names}
        unavailable = []
        for label, register, normalize in registers:
            try:
                values = self.arm.bus.sync_read(
                    register, motors=joint_names, normalize=normalize, num_retry=0,
                )
                for name in joint_names:
                    readings[name].append(f"{label}={float(values[name]):.3f}")
            except Exception as exc:
                unavailable.append(f"  {label}: unavailable ({type(exc).__name__}: {exc})")
        return [
            "Read-only motor snapshot after timeout (Goal_Position_deg in degrees; "
            "other registers in raw units):",
            *(f"  {name}: {', '.join(values)}" for name, values in readings.items()),
            *unavailable,
        ]

    def go_to_cartesian_pose(self, positions, orientations=None):
        previous_command = getattr(self, "_last_arm_command_deg", None)
        self._last_arm_command_deg = None
        observation = self.arm.get_observation()
        # LeRobot reads and commands the gripper in the same 0-100 units.
        # Capture once for this trajectory and reuse the value unchanged.
        gripper_position = float(observation["gripper.pos"])

        current_joints_deg = {
            name: float(observation[f"{name}.pos"])
            for name in self.arm_joint_names
        }
        
        # Nominal estimate for collision geometry: closed/open map to the
        # URDF limits. This is not a calibrated servo-to-URDF zero alignment.
        limits = self.motion_planner.scene.getJointInfo("gripper").limits
        lower, upper = limits.min_position[0], limits.max_position[0]
        current_joints_deg["gripper"] = float(np.rad2deg(
            lower + (gripper_position / 100.0) * (upper - lower)
        ))

        robot_positions = self.frida_to_robot(
            np.atleast_2d(np.asarray(positions, dtype=float))
        )

        # Only continue a successfully completed command while feedback remains
        # nearby. After manual repositioning or a failure, plan from measurement.
        planning_options = {}
        if previous_command is not None and all(
            np.isfinite(current_joints_deg[name])
            and abs(current_joints_deg[name] - previous_command[name]) <= 2.0
            for name in self.arm_joint_names
        ):
            planning_options["previous_command_deg"] = previous_command
        trajectory = self.motion_planner.plan_xyz_path(
            robot_positions, current_joints_deg, **planning_options
        )
        if trajectory is None:
            self._last_arm_command_deg = planning_options.get("previous_command_deg")
            return True

        commands = []

        for q_model in trajectory.positions:
            # Invert the same startup calibration used by plan_xyz_path:
            # model_deg = observation_deg + offset_deg.
            action = {
                f"{name}.pos": (
                    float(np.rad2deg(angle))
                    - self.motion_planner.joint_offsets_deg.get(name, 0.0)
                )
                for name, angle in zip(trajectory.joint_names, q_model)
            }
            action["gripper.pos"] = gripper_position
            commands.append(action)

        self.execute_trajectory(trajectory.times, commands)
            
    def get_cartesian_pose(self):
        """Return the measured brush-tip [x, y, z] in FRIDA coordinates, meters."""
        observation = self.arm.get_observation()
        planner = self.motion_planner
        q = planner.scene.getCurrentJointPositions().copy()
        for name in planner.joint_names:
            angle = float(observation[f"{name}.pos"])
            if not np.isfinite(angle):
                raise ValueError(f"Invalid angle for {name}")
            index = planner.scene.getJointPositionIndices([name])[0]
            q[index] = np.deg2rad(angle + planner.joint_offsets_deg[name])

        # The brush is fixed to the gripper body; jaw opening does not affect FK.
        pose = planner.scene.forwardKinematics(q, planner.tip_frame, planner.base_frame)
        # Inverse of frida_to_robot: robot [x, y, z] -> FRIDA [-y, x, z].
        return [-float(pose[1, 3]), float(pose[0, 3]), float(pose[2, 3])]
