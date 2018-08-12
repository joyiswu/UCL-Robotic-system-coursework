#!/usr/bin/env python
import PyKDL
import rospkg
import rosbag
import rospy
import math
import numpy as np
from numpy import *
from numpy import linalg as LA
from inverse_kinematics.YoubotKDL import YoubotKDL
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState
import time

# Global
Obstacle = []
check_point_list = []
angle = {}
frame_xyz = []


class c_class(YoubotKDL):
    def __init__(self):
        super(c_class, self).__init__()


#Obstacle contains 60 points information, and position is 3*5 vector
#and it contain 5 joint position, eta and rho size : 1*5
def force_repulsive(position, obstacle, eta, rho):

    # Calculate the minize distance
    J1 = position[0]
    J2 = position[1]
    J3 = position[2]
    J4 = position[3]
    J5 = position[4]

    min_norm_J1 = 1
    min_norm_J2 = 1
    min_norm_J3 = 1
    min_norm_J4 = 1
    min_norm_J5 = 1
    for i in range(len(obstacle)):

        norm_J1 = LA.norm((obstacle[i][0] - J1[0], obstacle[i][1] - J1[1], obstacle[i][2] - J1[2]))
        norm_J2 = LA.norm((obstacle[i][0] - J2[0], obstacle[i][1] - J2[1], obstacle[i][2] - J2[2]))
        norm_J3 = LA.norm((obstacle[i][0] - J3[0], obstacle[i][1] - J3[1], obstacle[i][2] - J3[2]))
        norm_J4 = LA.norm((obstacle[i][0] - J4[0], obstacle[i][1] - J4[1], obstacle[i][2] - J4[2]))
        norm_J5 = LA.norm((obstacle[i][0] - J5[0], obstacle[i][1] - J5[1], obstacle[i][2] - J5[2]))

        if norm_J1 < min_norm_J1:
            min_norm_J1 = norm_J1
            vect1 = [obstacle[i][0] - J1[0], obstacle[i][1] - J1[1], obstacle[i][2] - J1[2]]

        if norm_J2 < min_norm_J2:
            min_norm_J2 = norm_J2
            vect2 = [obstacle[i][0] - J2[0], obstacle[i][1] - J2[1], obstacle[i][2] - J2[2]]

        if norm_J3 < min_norm_J3:
            min_norm_J3 = norm_J3
            vect3 = [obstacle[i][0] - J3[0], obstacle[i][1] - J3[1], obstacle[i][2] - J3[2]]

        if norm_J4 < min_norm_J4:
            min_norm_J4 = norm_J4
            vect4 = [obstacle[i][0] - J4[0], obstacle[i][1] - J4[1], obstacle[i][2] - J4[2]]

        if norm_J5 < min_norm_J5:
            min_norm_J5 = norm_J5
            vect5 = [obstacle[i][0] - J5[0], obstacle[i][1] - J5[1], obstacle[i][2] - J5[2]]


    if min_norm_J1 <= rho[0]:
        Frep_J1 = (0, 0, 0)
    else:
        coefficient = eta[0] * (1/min_norm_J1 - 1/rho[0]) * 1/(min_norm_J1 * min_norm_J1) / min_norm_J1
        Frep_J1 = vect1
        [x * coefficient for x in Frep_J1]

    if min_norm_J2 <= rho[1]:
        Frep_J2 = (0, 0, 0)
    else:
        coefficient = eta[1] * (1/min_norm_J2 - 1/rho[1]) * 1/(min_norm_J2 * min_norm_J2) / min_norm_J2
        Frep_J2 = vect2
        [x * coefficient for x in Frep_J2]

    if min_norm_J3 <= rho[2]:
        Frep_J3 = (0, 0, 0)
    else:
        coefficient = eta[2] * (1/min_norm_J3 - 1/rho[2]) * 1/(min_norm_J3 * min_norm_J3) / min_norm_J3
        Frep_J3 = vect3
        [x * coefficient for x in Frep_J3]

    if min_norm_J4 <= rho[3]:
        Frep_J4 = (0, 0, 0)
    else:
        coefficient = eta[3] * (1/min_norm_J4 - 1/rho[3]) * 1/(min_norm_J4 * min_norm_J4) / min_norm_J4
        Frep_J4 = vect4
        [x * coefficient for x in Frep_J4]

    if min_norm_J5 <= rho[4]:
        Frep_J5 = (0, 0, 0)
    else:
        coefficient = eta[4] * (1/min_norm_J5 - 1/rho[4]) * 1/(min_norm_J5 * min_norm_J5) / min_norm_J5
        Frep_J5 = vect5
        [x * coefficient for x in Frep_J5]

    Frep = []
    Frep.extend((Frep_J1, Frep_J2, Frep_J3, Frep_J4, Frep_J5))

    return  Frep

def force_attractive(position, position_final, zeta, d, check_point_num):

    norm_J = {}
    Fatt = {}
    for i in range(5):
        norm_J[i] = LA.norm((position[i][0] - position_final[check_point_num][0][i], position[i][1] - position_final[check_point_num][1][i], position[i][2] - position_final[check_point_num][2][i]))
        if  norm_J[i] <= d[i]:
            Fatt[i] = [position[i][0] - position_final[check_point_num][0][i], position[i][1] - position_final[check_point_num][1][i], position[i][2] - position_final[check_point_num][2][i]]
            [x * (-zeta[i]) for x in Fatt[i]]
            # Fatt[i] = -zeta[i]  (position[i][0] - position_final[check_point_num][0][i], position[i][1] - position_final[check_point_num][1][i], position[i][2] - position_final[check_point_num][2][i])
        else:
            Fatt[i] = [position[i][0] - position_final[check_point_num][0][i], position[i][1] - position_final[check_point_num][1][i], position[i][2] - position_final[check_point_num][2][i]]
            [x * (d[i] * zeta[i] / norm_J[i]) for x in Fatt[i]]
            # Fatt[i] = d[i] * zeta[i] * (position[i][0] - position_final[check_point_num][0][i], position[i][1] - position_final[check_point_num][1][i], position[i][2] - position_final[check_point_num][2][i]) / norm_J[i]

    return Fatt

def cal_obstacle_cylinder(x, y, z, r, h):
    # top
    cylinder_obstacle = {}
    cylinder_obstacle[0] = (x + r, y, z + (1 / 2) * h)
    cylinder_obstacle[1] = (x, y + r, z + (1 / 2) * h)
    cylinder_obstacle[2] = (x - r, y, z + (1 / 2) * h)
    cylinder_obstacle[3] = (x, y - r, z + (1 / 2) * h)
    # middle
    cylinder_obstacle[4] = (x + r, y, z)
    cylinder_obstacle[5] = (x, y + r, z)
    cylinder_obstacle[6] = (x - r, y, z)
    cylinder_obstacle[7] = (x, y - r, z)
    # bottom
    cylinder_obstacle[8] = (x + r, y, z - (1 / 2) * h)
    cylinder_obstacle[9] = (x, y + r, z - (1 / 2) * h)
    cylinder_obstacle[10] = (x - r, y, z - (1 / 2) * h)
    cylinder_obstacle[11] = (x, y - r, z - (1 / 2) * h)

    return cylinder_obstacle

def cal_obstacle_box(x,y,z,w,d,h):

    box_obstacle = {}
    # top
    box_obstacle[0] = (x + (1 / 2) * w, y + (1 / 2) * d, z + (1 / 2) * h)
    box_obstacle[1] = (x + (1 / 2) * w, y - (1 / 2) * d, z + (1 / 2) * h)
    box_obstacle[2] = (x - (1 / 2) * w, y + (1 / 2) * d, z + (1 / 2) * h)
    box_obstacle[3] = (x - (1 / 2) * w, y - (1 / 2) * d, z + (1 / 2) * h)
    # middle
    box_obstacle[4] = (x + (1 / 2) * w, y + (1 / 2) * d, z)
    box_obstacle[5] = (x + (1 / 2) * w, y - (1 / 2) * d, z)
    box_obstacle[6] = (x - (1 / 2) * w, y + (1 / 2) * d, z)
    box_obstacle[7] = (x - (1 / 2) * w, y - (1 / 2) * d, z)
    # bottom
    box_obstacle[8] = (x + (1 / 2) * w, y + (1 / 2) * d, z - (1 / 2) * h)
    box_obstacle[9] = (x + (1 / 2) * w, y - (1 / 2) * d, z - (1 / 2) * h)
    box_obstacle[10] = (x - (1 / 2) * w, y + (1 / 2) * d, z - (1 / 2) * h)
    box_obstacle[11] = (x - (1 / 2) * w, y - (1 / 2) * d, z - (1 / 2) * h)

    return box_obstacle


def joint_state_callback(msg):
    global frame_xyz
    global check_point_num
    global count
    joint = []
    for i in range(0, 5):
        joint.append(msg.position[i])

    # Forward kinematic to convert theta to 5 Transforamtion Martix
    T = forward_kinematics(joint)

    # Calculate each joint Oi_q from T01 to T05
    Oi_q = {}
    for i in range(5):
        Oi_q[i] = [T[i][0, 3], T[i][1, 3], T[i][2, 3]]


    # Set repulsive force parameter
    rho = [0.1, 0.1, 1, 1, 1]
    eta = [1, 1, 0.1, 1, 1]
    # Calculate repulsive force
    Force_repulsive = force_repulsive(Oi_q, Obstacle, eta, rho)
    # print Force_repulsive

    # Set attractive force parameter
    zeta = [0.1, 0.1, 1, 1, 1]
    d = [1, 1, 0.1, 1, 1]

    # Calculate attractive force
    Force_attractive = force_attractive(Oi_q, frame_xyz, zeta, d, check_point_num)
    # print 'Force_attractive', Force_attractive

    jacobian = jacobian_o(joint)
    #print jacobian

    (torque, torque_mag) = compute_torque(Force_repulsive, Force_attractive, jacobian)

    # Gradient Descent parameter
    alpha = (0.1, 0.1, 0.1, 0.1, 0.1)
    tol = 0.05

    # Calculate new position
    torque = [torque[0,0], torque[0,1], torque[0,2], torque[0,3], torque[0,4]]
    [x * alpha / torque_mag for x in torque]
    new_q = np.matrix(joint) + np.matrix(torque)

    global angle
    q_final = list(angle[check_point_num])

    # Publish joint trajectory
    msg = JointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = joint_names
    point = JointTrajectoryPoint()
    point.positions = [new_q[0, 0], new_q[0, 1], new_q[0, 2], new_q[0, 3],
                       new_q[0, 4]]
    msg.points.append(point)
    traj_publisher.publish(msg)
    print msg

    qiqf_diff = (new_q[0, 0] - q_final[0],
                 new_q[0, 1] - q_final[1],
                 new_q[0, 2] - q_final[2],
                 new_q[0, 3] - q_final[3],
                 new_q[0, 4] - q_final[4]
                )
    qiqf_diff = LA.norm(qiqf_diff)


    if abs(qiqf_diff) < tol:
        check_point_num = check_point_num + 1
        count = 1
    else:
        count = count + 1
        # this check point is out of the workspace, so change attrative point
        if count > 1000:
            check_point_num = check_point_num + 1
            count = 1




    # print np.matrix(jacobian[0])
    # print np.matrix(Force_repulsive[0])
    # print Force_attractive
    # print '1', np.matrix(Force_repulsive[0])* np.matrix(jacobian[0])
    # print np.matrix(Force_attractive[0]) * np.matrix(jacobian[0])
    # print np.matrix(Force_repulsive[0])* np.matrix(jacobian[0]) + np.matrix(Force_attractive[0]) * np.matrix(jacobian[0])

def compute_torque(Force_repulsive, Force_attractive, J_o):

    torque = 0
    for i in range (0,5):
        torque = torque + np.matrix(Force_repulsive[0])* np.matrix(J_o[0]) + np.matrix(Force_attractive[0]) * np.matrix(J_o[0])

    torque_mag = LA.norm(torque)

    return (torque, torque_mag)

def jacobian_o(theta):

    dh_params =     [[0.033, pi / 2, 0.147, (170 * pi/ 180), -1],
                    [0.155, 0, 0, (65 * pi/ 180) + pi / 2, -1],
                    [0.135, 0, 0, (-146 * pi/ 180), -1],
                    [0.0, pi / 2, 0, (102.5 * pi/ 180) + pi / 2, -1],
                    [0, 0, 0.183, (167.5 * pi/ 180) + pi, -1]]
    a1 = dh_params[0][0]
    a2 = dh_params[1][0]
    a3 = dh_params[2][0]
    d1 = dh_params[0][2]
    d5 = dh_params[4][2]

    #From joint_state_callback
    theta1 = theta[0]
    theta2 = theta[1]
    theta3 = theta[2]
    theta4 = theta[3]
    theta5 = theta[4]

    #Jacobian oi, computed from respective T0i
    jo1_row1 = (-a1*sin(theta1), 0, 0, 0, 0)
    jo1_row2 = (a1*cos(theta1), 0, 0, 0, 0)
    jo1_row3 = (0, 0, 0, 0, 0)
    jo1 = (jo1_row1, jo1_row2, jo1_row3)
    jo1 = np.array(jo1)

    jo2_row1 = (-sin(theta1)*(a1+a2*cos(theta2)), -a2*cos(theta1)*sin(theta2) , 0, 0, 0)
    jo2_row2 = (cos(theta1)*(a1+a2*cos(theta2)), -a2*sin(theta1)*sin(theta2), 0, 0, 0)
    jo2_row3 = (0, a2*cos(theta2), 0, 0, 0)
    jo2 = (jo2_row1, jo2_row2, jo2_row3)

    jo3_row1 = (-sin(theta1)*(a1+a3*cos(theta2+theta3)+a2*cos(theta2)), -cos(theta1)*(a3*sin(theta2+theta3)+a2*sin(theta2)), -a3*sin(theta2+theta3)*cos(theta1), 0, 0)
    jo3_row2 = (cos(theta1)*(a1+a3*cos(theta2+theta3)+a2*cos(theta2)), -sin(theta1)*(a3*sin(theta2+theta3)+a2*sin(theta2)), -a3*sin(theta2+theta3)*sin(theta1), 0, 0)
    jo3_row3 = (0, a3*cos(theta2+theta3)+a2*cos(theta2), a3*cos(theta2+theta3), 0, 0)
    jo3 = (jo3_row1, jo3_row2, jo3_row3)

    jo4_row1 = (-sin(theta1)*(a1+a3*cos(theta2+theta3)+a2*cos(theta2)), -cos(theta1)*(a3*sin(theta2+theta3)+a2*sin(theta2)), -a3*sin(theta2+theta3)*cos(theta1), 0, 0)
    jo4_row2 = (cos(theta1)*(a1+a3*cos(theta2+theta3)+a2*cos(theta2)), -sin(theta1)*(a3*sin(theta2+theta3)+a2*sin(theta2)),  -a3*sin(theta2+theta3)*sin(theta1), 0, 0)
    jo4_row3 = (0, a3*cos(theta2+theta3)+a2*cos(theta2), a3*cos(theta2+theta3), 0, 0)
    jo4 = (jo4_row1, jo4_row2, jo4_row3)

    jo5_row1 = (-sin(theta1)*(a1+a3*cos(theta2+theta3)+a2*cos(theta2)+d5*sin(theta2+theta3+theta4)), cos(theta1)*(a3*sin(theta2+theta3)+a2*sin(theta2)-d5*cos(theta2+theta3+theta4)),
                -cos(theta1)*(a3*sin(theta2+theta3)-d5*cos(theta2+theta3+theta4)), d5*cos(theta2+theta3+theta4)*cos(theta1), 0)
    jo5_row2 = (cos(theta1)*(a1+a3*cos(theta2+theta3)+a2*cos(theta2)+d5*sin(theta2+theta3+theta4)), -sin(theta1)*(a3*sin(theta2+theta3)+a2*sin(theta2)-d5*cos(theta2+theta3+theta4)),
                -sin(theta1)*(a3*sin(theta2+theta3)-d5*cos(theta2+theta3+theta4)), d5*cos(theta2+theta3+theta4)*sin(theta1), 0)
    jo5_row3 = (0, a3*cos(theta2+theta3)+a2*cos(theta2)+d5*sin(theta2+theta3+theta4), a3*cos(theta2+theta3)+d5*sin(theta2+theta3+theta4), d5*sin(theta2+theta3+theta4), 0)
    jo5 = (jo5_row1, jo5_row2, jo5_row3)

    return (jo1, jo2, jo3, jo4, jo5)

def forward_kinematics(joint):
    # Defining DH Parameters
    # youbot_kinematics = YoubotKinematics()
    youbot_martix = [[0.033, pi / 2, 0.147, (170 * pi/ 180), -1],
                    [0.155, 0, 0, (65 * pi/ 180) + pi / 2, -1],
                    [0.135, 0, 0, (-146 * pi/ 180), -1],
                    [0.0, pi / 2, 0, (102.5 * pi/ 180) + pi / 2, -1],
                    [0, 0, 0.183, (167.5 * pi/ 180) + pi, -1]]
    a = []
    alpha = []
    d = []
    q = []
    for i in range(5):
        a.append(youbot_martix[i][0])
        alpha.append(youbot_martix[i][1])
        d.append(youbot_martix[i][2])
        q.append(youbot_martix[i][3] - joint[i])

    # Creating individual transformation matrices
    T = {}
    for i in range(5):
        T[i] = np.matrix([[cos(q[i]), -sin(q[i])*cos(alpha[i]), sin(q[i])*sin(alpha[i]), a[i]*cos(q[i])],
                       [sin(q[i]), cos(q[i])*cos(alpha[i]), -cos(q[i])*sin(alpha[i]), a[i]*sin(q[i])],
                       [0, sin(alpha[i]), cos(alpha[i]), d[i]],
                       [0, 0, 0, 1]])

    # Matrix Transformations
    T[1] = T[0]*T[1]                  # T0_2
    T[2] = T[1]*T[2]                  # T0_3
    T[3] = T[2]*T[3]                  # T0_4
    T[4] = T[3]*T[4]                  # T0_5
    Pose = [T[0], T[1], T[2], T[3], T[4]]
    return Pose


def main():
    rospy.init_node('path_planning_node')
    global joint_names
    global traj_publisher
    global check_point_num
    check_point_num = 0
    global count
    count = 1
    # joint_names = rospy.get_param('/EffortJointInterface_trajectory_controller/joints')
    # num = 1
    #
    # traj_publisher = rospy.Publisher('/EffortJointInterface_trajectory_controller/command', JointTrajectory,
    #                                  queue_size=3)
    #
    # subscriber_joint_state_ = rospy.Subscriber('/joint_states', JointState, joint_state_callback,
    #                                                 queue_size=5)

    # subscriber_link_states = rospy.Subscriber('/gazebo/link_states', LinkStates,
    #                                           queue_size=5)
    # print subscriber_link_states


    ybKDL = YoubotKDL()

    b = c_class()

    ##TODO: Write a code to extract data from your bag.
    bag = rosbag.Bag('/home/joy/catkin_ws/src/compgx01_lab/cw2_helper/bags/data_q4d.bag')

    x = []
    y = []
    z = []

    for topic, msg, t in bag.read_messages(topics=['target_position']):  # geometry_msgs/TransformStamped, 10 msgs

        # x.append(msg.x)
        # y.append(msg.y)
        # z.append(msg.z)
        print msg
    bag.close()
    #
    # # Compute the joint angles to reach for each joint for each check point
    # global angle
    # desired_pose = PyKDL.Frame()
    #
    # for i in range(0, 9):
    #     Rot_K = PyKDL.Rotation(1, 0, 0,
    #                            0, 1, 0,
    #                            0, 0, 1)
    #     Trans_K = PyKDL.Vector(x[i], y[i], z[i])
    #
    #     desired_pose.M = Rot_K  # Assigns rotation matrix to frame rotation
    #     desired_pose.p = Trans_K  # Assigns translation to frame vector
    #
    #     angle[i] = ybKDL.inverse_kinematics_closed(desired_pose)
    #
    # print angle
    # # Compute and translate the joint angles to XYZ positions to reach using forward kinematics
    # x_i = []
    # y_i = []
    # z_i = []
    # global frame_xyz
    # for i in range(9):
    #     T = forward_kinematics(angle[i])
    #     x_i.append(((T[0])[0, 3], (T[1])[0, 3], (T[2])[0, 3], (T[3])[0, 3], (T[4])[0, 3]))
    #     y_i.append(((T[0])[1, 3], (T[1])[1, 3], (T[2])[1, 3], (T[3])[1, 3], (T[4])[1, 3]))
    #     z_i.append(((T[0])[2, 3], (T[1])[2, 3], (T[2])[2, 3], (T[3])[2, 3], (T[4])[2, 3]))
    #     frame_xyz.append((x_i[0], y_i[0], z_i[0]))
    #
    # # print frame_xyz[0][0][1]
    #
    # # Choose 12 points for each obstacle and calculate the position of it
    # box0_obstacle = cal_obstacle_box(0.00051985828679, -0.202068169352, 0.0024647590529, 1, 1, 0.069438)
    # box1_obstacle = cal_obstacle_box(0.00055368633152, -0.202071341584, 0.128464745519, 0.194033, 0.077038, 0.198110)
    # box2_obstacle = cal_obstacle_box(-0.0313762478077, -0.210387355195, 0.14747316014, 0.167065, 0.112100, 0.04652)
    #
    # cylinder0_obstacle = cal_obstacle_cylinder(0.0363756916606, -0.192753658385, 0.285756418212, 0.05, 0.08)
    # cylinder1_obstacle = cal_obstacle_cylinder(-0.0942696237243, -0.226770085256, 0.285835338377, 0.06, 0.14)
    #
    # # Store all the Obstacle data (12 for each) 60 total
    # global Obstacle
    # for i in range(12):
    #     Obstacle.append(box0_obstacle[i])
    #     Obstacle.append(box1_obstacle[i])
    #     Obstacle.append(box2_obstacle[i])
    #     Obstacle.append(cylinder0_obstacle[i])
    #     Obstacle.append(cylinder1_obstacle[i])


    rospy.spin()
    time.sleep(1)


    raw_input('Press enter to rerun the trajectory\n')

if __name__ == '__main__':
    main()
