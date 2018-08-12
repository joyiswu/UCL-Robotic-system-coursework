#!/usr/bin/env python
import PyKDL
import rospkg
import rosbag
import rospy
import math
import numpy as np
from numpy import *
from scipy.linalg import expm, logm
from inverse_kinematics.YoubotKDL import YoubotKDL
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class b_class(YoubotKDL):
    def __init__(self):
        super(b_class, self).__init__()


def main():
    rospy.init_node('path_planning_node')
    joint_names = rospy.get_param('/EffortJointInterface_trajectory_controller/joints')
    traj_publisher = rospy.Publisher('/EffortJointInterface_trajectory_controller/command', JointTrajectory,
                                     queue_size=3)
    ybKDL = YoubotKDL()

    b = b_class()

    ##TODO: Write a code to extract data from your bag.
    bag = rosbag.Bag('/home/joy/catkin_ws/src/compgx01_lab/cw2_helper/bags/data_q4b.bag')
    time_list = []
    quaternion_list = []
    translation_list = []

    # Store position velocity and time information
    for topic, msg, t in bag.read_messages():

        time_list.append(msg.header.stamp.secs+float(msg.header.stamp.nsecs)/1000000000)
        quaternion_list.append((msg.transform.rotation.x,msg.transform.rotation.y,msg.transform.rotation.z,msg.transform.rotation.w))
        translation_list.append((msg.transform.translation.x,msg.transform.translation.y,msg.transform.translation.z))
        # position_list.append(msg.position)
        # velocity_list.append(msg.velocity)



    T = {}
    # Calculate Transfrom Mratix
    for i in range(len(time_list)):
        # Convert Quaternion to Rotation Martix
        R_11 = 1 - 2 * pow(quaternion_list[i][1], 2) - 2 * pow(quaternion_list[i][2], 2)
        R_12 = 2 * quaternion_list[i][0] * quaternion_list[i][1] - 2 * quaternion_list[i][2] * quaternion_list[i][3]
        R_13 = 2 * quaternion_list[i][0] * quaternion_list[i][2] + 2 * quaternion_list[i][1] * quaternion_list[i][3]
        R_21 = 2 * quaternion_list[i][0] * quaternion_list[i][1] + 2 * quaternion_list[i][2] * quaternion_list[i][3]
        R_22 = 1 - 2 * pow(quaternion_list[i][0], 2) - 2 * pow(quaternion_list[i][2], 2)
        R_23 = 2 * quaternion_list[i][1] * quaternion_list[i][2] - 2 * quaternion_list[i][0] * quaternion_list[i][3]
        R_31 = 2 * quaternion_list[i][0] * quaternion_list[i][2] - 2 * quaternion_list[i][1] * quaternion_list[i][3]
        R_32 = 2 * quaternion_list[i][1] * quaternion_list[i][2] + 2 * quaternion_list[i][0] * quaternion_list[i][3]
        R_33 = 1 - 2 * pow(quaternion_list[i][0], 2) - 2 * pow(quaternion_list[i][1],2 )

        # Combine rotation martix and postion to get Transform Martix
        T[i] = ((R_11, R_12, R_13, translation_list[i][0]),
                (R_21, R_22, R_23, translation_list[i][1]),
                (R_31, R_32, R_33, translation_list[i][2]),
                (0, 0, 0, 1))




    # Adding points during each time slot for transform each second
    ##TODO: Write a code to create your trajectory from bag data.
    new_time_list = []
    new_T_list = []
    new_time_list.append(time_list[0])
    new_T_list.append(T[0])

    # Calculate Straight line Paths
    for i in range(len(time_list) - 1):
        time = 0

        # Assign inital martix and final martix
        T_initial = T[i]
        T_final = T[i + 1]
        delta_time = (time_list[i + 1] - time_list[i]) / 5
        # for j in range(1,int(time_list[i + 1] - time_list[i])+1):
        #
        #     T_initial = np.array(T_initial)
        #     T_final = np.array(T_final)
        #     inv_T_initial = np.linalg.inv(T_initial)
        #     log_martix = logm(np.dot(inv_T_initial, T_final))
        #     exp_martix = expm(log_martix)
        #     result_martix = np.dot(T_initial, exp_martix)
        #     new_T_list.append(result_martix)
        #     T_initial = result_martix
        #     time = time_list[i]+j
        #     new_time_list.append(time)

        for j in range(1,6):
            new_time_list.append(time_list[i] + j * delta_time)
            time = j * 0.2
            T_initial = np.array(T_initial)
            T_final = np.array(T_final)
            inv_T_initial = np.linalg.inv(T_initial)
            log_martix = logm(np.dot(inv_T_initial, T_final) * time)
            exp_martix = expm(log_martix)
            result_martix = np.dot(T_initial, exp_martix)
            new_T_list.append(result_martix)

    # the last time and transform Martix
    # new_time_list.append(time_list[-1])
    # new_T_list.append(T[len(time_list)-1])

    # Convert to position via KDL inverse kinematics function
    pose = PyKDL.Frame()
    result_position = {}
    for i in range(len(new_time_list)):
        rotmat = PyKDL.Rotation(new_T_list[i][0][0], new_T_list[i][0][1], new_T_list[i][0][2],
                                new_T_list[i][1][0], new_T_list[i][1][1], new_T_list[i][1][2],
                                new_T_list[i][2][0], new_T_list[i][2][1], new_T_list[i][2][2])
        position = PyKDL.Vector(new_T_list[i][0][3], new_T_list[i][1][3], new_T_list[i][2][3])

        pose.M = rotmat
        pose.p = position
        result_position[i] = ybKDL.inverse_kinematics_closed(pose)
        #print result_position


    # Splict the time info to secs and nsecs
    result_secs_list = []
    result_nsecs_list = []
    for i in range(len(new_time_list)):
        result_nsecs_list.append(int(new_time_list[i]%1*1000000000))
        result_secs_list.append(int(new_time_list[i]-(new_time_list[i]%1)))

    print time_list
    print new_time_list
    # print result_nsecs_list
    # print result_secs_list
    # print new_T_list
    # # print T
    print len(new_T_list),len(new_time_list)


   # print inkmk_kdl
   #  print result_position
   #  print time_list
   #  print len(new_time_list)
   #  print len(new_T_list)
   #  print result_secs_list
   #  print result_nsecs_list

    # Publish joint trajectory
    msg = JointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = joint_names
    for i in range(len(new_time_list)):
        point = JointTrajectoryPoint()
        point.positions = [result_position[i][0],result_position[i][1],result_position[i][2],result_position[i][3],result_position[i][4]]
        point.time_from_start.nsecs = result_nsecs_list[i]
        point.time_from_start.secs = result_secs_list[i]

        msg.points.append(point)
    # print msg
    traj_publisher.publish(msg)

    # print msg
    rospy.sleep(2)


    raw_input('Press enter to rerun the trajectory\n')

if __name__ == '__main__':
    main()
