#!/usr/bin/env python

import rospy
from math import pi
from math import pi, sin, cos, atan, atan2, acos, asin
from sympy import symbols, simplify, Matrix
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, Pose
import tf2_ros
import numpy as np
from scipy import linalg as LA
from inverse_kinematics.YoubotKinematics import YoubotKinematics
from inverse_kinematics.YoubotKDL import YoubotKDL
import PyKDL
import time


class YoubotTemplate(YoubotKinematics):
    def __init__(self):
        super(YoubotTemplate, self).__init__()
        self.subscriber_joint_state_ = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback,
                                                        queue_size=5)

    def joint_state_callback(self, msg):
        theta = []
        joint = []
        for i in range(0, 5):
            theta.append(self.dh_params[i][3] - msg.position[i])
            joint.append(msg.position[i])

        pose = np.matrix([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.forward_kinematics(joint)
        self.inverse_kinematics_jac(pose,theta)
        return theta

    def forward_kinematics(self, joint):
        # Defining DH Parameters
        # youbot_kinematics = YoubotKinematics()
        youbot_martix = self.dh_params

        a = []
        alpha = []
        d = []
        q = []
        for i in range(5):
            a.append(youbot_martix[i][0])
            alpha.append(youbot_martix[i][1])
            d.append(youbot_martix[i][2])
            q.append(youbot_martix[i][3] - joint[i])
            #q.append(joint[i])

        # print 'q:', q

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


    def broadcast_pose(self, pose):
        trans = TransformStamped()

        # Fill in trans.transform with the pose you got from fkine

        trans.header.frame_id = "arm_link_0"
        trans.header.stamp = rospy.Time.now()
        trans.child_frame_id = "arm_end_effector"

        self.pose_broadcaster.sendTransform(trans)

    def get_jacobian(self, joint):
        # Derivation of Jacobian Angular Elements
        T0_0 = np.matrix([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        z = {}
        o = {}
        z[0] = np.matrix(T0_0[[0, 1, 2], 2])
        o[0] = np.matrix(T0_0[[0, 1, 2], 3])

        # Derivation of Jacobian Linear Elements
        for i in range(5):
            z[i+1] = np.matrix(joint[i][[0, 1, 2], 2])
            o[i+1] = np.matrix(joint[i][[0, 1, 2], 3])

        Jv = {}
        J = {}
        # Vector cross-product.
        for i in range(5):
            Jv[i] = np.cross(z[i].T, (o[5] - o[i]).T)

        # Derivation of Jacobian Elements
        for i in range(5):
            J[i] = np.concatenate((Jv[i].T, z[i]), axis=0)

        #print 'J', np.concatenate((J[0], J[1], J[2], J[3], J[4]), axis=1)
        return np.concatenate((J[0], J[1], J[2], J[3], J[4]), axis=1)




    def inverse_kinematics_closed(self, desired_pose):
        a1 = 0.033
        a2 = 0.155
        a3 = 0.135
        d1 = 0.147
        d5 = 0.183
        r11 = desired_pose[0][0]
        r12 = desired_pose[0][1]
        r13 = desired_pose[0][2]
        r21 = desired_pose[1][0]
        r22 = desired_pose[1][1]
        r23 = desired_pose[1][2]
        r31 = desired_pose[2][0]
        r32 = desired_pose[2][1]
        r33 = desired_pose[2][2]

        x = desired_pose[0][3]
        y = desired_pose[1][3]
        z = desired_pose[2][3]

        theta1a = atan(y/x)
        theta1 = []
        theta2 = []
        theta3 = []
        theta4 = []
        theta5 = []
        solution1 = []
        solution2 = []
        solution3 = []
        solution4 = []

        if (theta1a+pi>(169*pi/180) and theta1a-pi<(-169*pi/180)):

            theta1 = theta1a

        else:

            theta1.extend((theta1a,theta1a+pi))

        if theta1 == theta1a:
            theta5 = ((atan2(((r21*cos(theta1))-(r11*sin(theta1))),((r22*cos(theta1))-(r12*sin(theta1))))))
            theta3a = acos(((z-d1-d5*r33)**2+(-y/sin(theta1)+a1+d5*r23/sin(theta1))**2-(a2*a2)-(a3*a3))/(2*a2*a3))
            theta3.extend((theta3a,-theta3a))

            for j in range(0,2):

                theta2.append((atan2(((-y/sin(theta1))+a1+((d5*r23)/sin(theta1))),(z-d1-d5*r33))-atan2(a3*sin(theta3[j]),(a2+a3*cos(theta3[j])))))
                theta4.append(((atan2((r32/-sin(theta5)),r33)-theta2[j]-theta3[j])))

            solution1.extend((theta1, theta2[0], theta3[0], theta4[0], theta5))
            solution2.extend((theta1,theta2[1],theta3[1],theta4[1],theta5))

            return (solution1, solution2)

        else:

            for i in range(0,2):
                theta5.append((atan2(((r21*cos(theta1[i]))-(r11*sin(theta1[i]))), ((r22*cos(theta1[i])) - (r12 * sin(theta1[i]))))))
                theta3a = acos(round((((z-d1-d5*r33)**2+(-y/sin(theta1[i])+a1+d5*r23/sin(theta1[i]))**2-(a2**2)-(a3**2))/(2*a2*a3)),4))
                theta3a = round(theta3a)

                theta3.extend((theta3a,-theta3a))

            for j in range(0,2):
                b = theta1[j]
                c = theta5[j]
                theta2 = atan2(((-y/sin(b))+a1+((d5*r23)/sin(b))), (z-d1-d5*r33))-atan2(a3*sin(theta3[j+1]), (a2+a3*cos(theta3[j+1])))
                theta4 = atan2((r32/-sin(c)), r33)-theta2[j]-theta3[j+1]
                theta2.append(theta2[j])
                theta4.append(theta4[j])

            for k in range(0, 2):
                b = theta1[k]
                c = theta5[k]
                theta2 = atan2(((-y/sin(b))+a1 +((d5*r23)/sin(b))), (z-d1-d5*r33))-atan2(a3*sin(theta3[k+1]), (a2+a3*cos(theta3[k+1])))
                theta4 = atan2((r32/-sin(c)), r33) - theta2[k] - theta3[k+1]
                theta2.append(theta2[k])
                theta4.append(theta4[k])

            solution1.extend((theta1[0], theta2[0], theta3[0], theta4[0], theta5[0]))
            solution2.extend((theta1[0], theta2[2], theta3[1], theta4[2], theta5[0]))
            solution3.extend((theta1[1], theta2[1], theta3[2], theta4[1], theta5[1]))
            solution4.extend((theta1[1], theta2[3], theta3[3], theta4[3], theta5[1]))

            return (solution1, solution2, solution3, solution4)

    def inverse_kinematics_jac(self, desired_pose, current_joint):

        # Define the desired pose and lamda constant, H
        pose_star = desired_pose  # setting desired pose to pose star
        H = 0.01

        for k in range(0, 10000):

            if k == 0:

                theta = []

                for i in range(0, 5):
                    theta.append(current_joint[i])
                # Perform Forward Kinematics Using Current Joint Theta Values
                TM = self.forward_kinematics(
                    theta)  # TM consists of all the transformation matrices from forward kinematics
                T5 = TM[4]  # choose T0_5

                print T5

                # Compute Jacobian Matrix
                J = self.get_jacobian(TM)

                # (Q3e) Check for singularity
                JT = np.transpose(J)
                D = LA.det(J*JT)

                if D == 0:
                    print 'Singularity'
                    noise = 1.003
                    theta = [noise * x for x in theta]  # Add a bit of noise to theta to move it out of singularity
                    k = k + 1
                    continue

            elif k > 0:

                theta = new_theta

                # Perform Forward Kinematics Using Current Joint Theta Values
                TM = self.forward_kinematics(
                    theta)  # TM consists of all the transformation matrices from forward kinematics
                T5 = TM[4]  # choose T0_5

                # Compute Jacobian Matrix
                J = self.get_jacobian(TM)  # get jacobian matrix from jacobian function above
                # (Q3e) Check for singularity
                JT = np.transpose(J)
                D = LA.det(J*JT)

                if D == 0:
                    print 'Singularity'
                    noise = 1.003
                    theta = [noise * x for x in theta]  # Add a bit of noise to theta to move it out of singularity
                    k = k + 1
                    continue

            # Rotation Matrix Elements
            r11 = T5[0, 0]
            r12 = T5[0, 1]
            r13 = T5[0, 2]
            x = T5[0, 3]
            r21 = T5[1, 0]
            r22 = T5[1, 1]
            r23 = T5[1, 2]
            y = T5[1, 3]
            r31 = T5[2, 0]
            r32 = T5[2, 1]
            r33 = T5[2, 2]
            z = T5[2, 3]

            # Convert Rotation Matrix to Angle-Axis Representation
            q = acos((r11 + r22 + r33 - 1) / 2)  # calculate theta for axis-angle
            rx = (1 / (2 * sin(q))) * (r32 - r23)
            ry = (1 / (2 * sin(q))) * (r13 - r31)
            rz = (1 / (2 * sin(q))) * (r21 - r12)
            p = []
            p.extend((x, y, z, rx, ry, rz))

            # Iterations
            new_theta = np.matrix(theta).T + H * J.T * ((pose_star - np.matrix(p)).T)  # Perform Iterations until solutionution converges
            diff = LA.norm(
                new_theta.T - np.matrix(theta))  # calculate difference between current theta vector magnitude and previous theta vector magnitude

            if diff > 0.008:
                new_theta = new_theta.T
                new_theta = new_theta.tolist()[0]  # force new set of theta from matrix back to list
                k = k + 1
                break

            else:
                print 'final theta'
                print new_theta

        return new_theta

def main():

    rospy.init_node('CourseWork2')
    youbot_kdl = YoubotKDL()
    youbot_template = YoubotTemplate()

    # theta = [pi, pi, pi, pi, 0]
    # forward_martix = youbot_template.forward_kinematics(theta)
    # jocbian_martix = youbot_template.get_jacobian(forward_martix)
    # kdl_jacobian = youbot_kdl.get_jacobian(youbot_kdl.current_joint_position)
    #
    # inverse_kinematics_closed_result = youbot_template.inverse_kinematics_closed(np.array(forward_martix[-1]))

    # print youbot_kdl.current_joint_position
    # # print jacobian

    # Check 3a Jacobian answer via KDL
    # print forward_martix[-1]
    # print 'inverse_kinematics_closed_result:', inverse_kinematics_closed_result
    # print 'forward kinematic martix:', forward_martix
    # print 'jacobian martix:', jocbian_martix
    # print 'kdl jacbian:', kdl_jacobian
    # print youbot_martix

    while not rospy.is_shutdown():
        rospy.spin()
        time.sleep(1)




if __name__ == '__main__':
    main()
