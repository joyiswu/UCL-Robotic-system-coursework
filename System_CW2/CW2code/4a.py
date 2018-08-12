#!/usr/bin/env python

import rospkg
import rosbag
import rospy
import math
from numpy import *
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


if __name__ == '__main__':
    rospy.init_node('path_planning_node')
    joint_names = rospy.get_param('/EffortJointInterface_trajectory_controller/joints')
    #rospack = rospkg.RosPack()
    traj_publisher = rospy.Publisher('/EffortJointInterface_trajectory_controller/command', JointTrajectory,
                                     queue_size=3)


    ##TODO: Write a code to extract data from your bag.
    bag = rosbag.Bag('/home/joy/catkin_ws/src/compgx01_lab/cw2_helper/bags/data_q4a.bag')
    time_list = []
    position_list = []
    velocity_list = []
    Cubic_martix = []

    # Store position velocity and time information
    for topic, msg, t in bag.read_messages():

        time_list.append(msg.header.stamp.secs+float(msg.header.stamp.nsecs)/1000000000)
        position_list.append(msg.position)
        velocity_list.append(msg.velocity)
    print time_list, position_list, velocity_list
    ##TODO: Write a code to create your trajectory from bag data.

    A_J1 = []
    A_J2 = []
    A_J3 = []
    A_J4 = []
    A_J5 = []

    # Cubic Polynomial
    for index in range(0,9):
        position_velocity_row = []
        # Calculate Cublic martix
        # Cublic_line1 = (1,time_list[index],pow(time_list[index],2),pow(time_list[index],3))
        # Cublic_line2 = (0,1,2*time_list[index],3*pow(time_list[index],2))
        # Cublic_line3 = (1,time_list[index+1],pow(time_list[index+1],2),pow(time_list[index+1],3))
        # Cublic_line4 = (0,1,2*time_list[index+1],3*pow(time_list[index+1],2))
        # Cubic_martix = (Cublic_line1,Cublic_line2,Cublic_line3,Cublic_line4)

        # The initial times are always 0 for each martix and final times are below
        time_final = time_list[index+1]-time_list[index]
        Cublic_line1 = (1, 0, 0, 0)
        Cublic_line2 = (0, 1, 0, 0)
        Cublic_line3 = (1, time_final, pow(time_final, 2), pow(time_final, 3))
        Cublic_line4 = (0, 1, 2*time_final, 3*pow(time_final, 2))
        Cubic_martix = (Cublic_line1, Cublic_line2, Cublic_line3, Cublic_line4)

        # Convert the value to martix
        Cubic_martix = np.array(Cubic_martix)

        # Calculate inv martix to calculate a martix
        inv_Cubic_martix = np.linalg.inv(Cubic_martix)

        for joint_num in range(5):
            position_velocity_row.append(np.array((position_list[index][joint_num],velocity_list[index][joint_num],position_list[index+1][joint_num],velocity_list[index+1][joint_num])))

        # Calculate a0,a1,a2,a3 for 5 joint and store as vector A
        A_J1.append(np.dot(inv_Cubic_martix,position_velocity_row[0]))
        A_J2.append(np.dot(inv_Cubic_martix,position_velocity_row[1]))
        A_J3.append(np.dot(inv_Cubic_martix,position_velocity_row[2]))
        A_J4.append(np.dot(inv_Cubic_martix,position_velocity_row[3]))
        A_J5.append(np.dot(inv_Cubic_martix,position_velocity_row[4]))


    # print A_J1
    # print A_J2
    # print A_J3
    # print A_J4
    # print A_J5
    # Adding 4 point during each time slot for 5 joints
    new_time_list = []
    new_position_list = []
    new_velocity_list = []
    new_time_list.append(time_list[0])
    new_position_list.append((position_list[0][0], position_list[0][1], position_list[0][2], position_list[0][3], position_list[0][4]))
    new_velocity_list.append((velocity_list[0][0], velocity_list[0][1], velocity_list[0][2], velocity_list[0][3], velocity_list[0][4]))
    for index in range(len(time_list) - 1):
        # divide each time slot as 5 part and get delta time to add 4 point in each slot
        delta_time = (time_list[index+1]-time_list[index])/5
        time = time_list[index]
        initial_time = 0
        for time_num in range(1,6):
            initial_time = time_num * delta_time
            joint_position_list = (1 * A_J1[index][0] + initial_time * A_J1[index][1] + pow(initial_time, 2) * A_J1[index][2] + pow(initial_time, 3) * A_J1[index][3],
                                   1 * A_J2[index][0] + initial_time * A_J2[index][1] + pow(initial_time, 2) * A_J2[index][2] + pow(initial_time, 3) * A_J2[index][3],
                                   1 * A_J3[index][0] + initial_time * A_J3[index][1] + pow(initial_time, 2) * A_J3[index][2] + pow(initial_time, 3) * A_J3[index][3],
                                   1 * A_J4[index][0] + initial_time * A_J4[index][1] + pow(initial_time, 2) * A_J4[index][2] + pow(initial_time, 3) * A_J4[index][3],
                                   1 * A_J5[index][0] + initial_time * A_J5[index][1] + pow(initial_time, 2) * A_J5[index][2] + pow(initial_time, 3) * A_J5[index][3])

            joint_velocity_list =(A_J1[index][1]+2*initial_time*A_J1[index][2]+3*pow(initial_time,2)*A_J1[index][3],
                                  A_J2[index][1]+2*initial_time*A_J2[index][2]+3*pow(initial_time,2)*A_J2[index][3],
                                  A_J3[index][1]+2*initial_time*A_J3[index][2]+3*pow(initial_time,2)*A_J3[index][3],
                                  A_J4[index][1]+2*initial_time*A_J4[index][2]+3*pow(initial_time,2)*A_J4[index][3],
                                  A_J5[index][1]+2*initial_time*A_J5[index][2]+3*pow(initial_time,2)*A_J5[index][3])
            # Calculate position
            new_position_list.append(joint_position_list)
            # Calculate velocity
            new_velocity_list.append(joint_velocity_list)
            new_time_list.append(time+time_num*delta_time)

    # Splict the time info to secs and nsecs
    result_secs_list = []
    result_nsecs_list = []
    for i in range(len(new_time_list)):
        result_nsecs_list.append(int(new_time_list[i]%1*1000000000))
        result_secs_list.append(int(new_time_list[i]-(new_time_list[i]%1)))

    #
    # print result_nsecs_list
    # print result_secs_list
    # print 'position_list:',position_list
    # print 'new_position_list:',new_position_list
    # print 'velocity_list:',velocity_list
    # print 'new_velocity_list:',new_velocity_list
    # print 'new time:',new_time_list
    #print joint_names



    # print "cublic" ,Cubic_martix
    # print "cublic", position_velocity_row
    # print A_J1

    # Publish joint trajectory
    msg = JointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = joint_names
    for i in range(len(new_time_list)):
        point = JointTrajectoryPoint()
        point.positions = new_position_list[i]
        point.velocities = new_velocity_list[i]
        point.time_from_start.nsecs = result_nsecs_list[i]
        point.time_from_start.secs = result_secs_list[i]

        msg.points.append(point)

    rospy.sleep(2)
    traj_publisher.publish(msg)




    raw_input('Press enter to rerun the trajectory\n')


