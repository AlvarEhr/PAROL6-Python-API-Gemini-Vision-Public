from Headless.robot_api import *

import time

def wave_command():

    current_joint_angles = get_robot_joint_angles()

    move_robot_joints([current_joint_angles[0], -90, 220, 0, 0, 180], speed_percentage=50)

    #jog_robot_joint(0,30,2.0)
    #jog_robot_joint((0+6),30,2.0)

    jog_multiple_joints([0,3,5],[70,40,60],0.6)

    dirty_timer = 3

    while(dirty_timer > 0):

        jog_multiple_joints([(0+6),(3+6),(5+6)],[70,40,60],1.2)

        jog_multiple_joints([0,3,5],[70,40,60],1.2)

        dirty_timer -= 1

    move_robot_joints(current_joint_angles, speed_percentage=50)

def search(search_pattern = "sweep", position = "low", continue_from_current_position = False):

    if search_pattern == "sweep":

        start_joint_angles = [120, -90, 200, 0, -90, 180] 
        
        if position == "mid": 
            start_joint_angles = [120, -93.9459375, 197.5112171052632, 0.0, -68.39375, 180.039375]
        elif position == "high":
            start_joint_angles = [120, -93.9459375, 197.5112171052632, 0.0, -40.39375, 180.039375]

        end_joint_angles = start_joint_angles.copy()
        end_joint_angles[0] = -30

        if not continue_from_current_position:

            move_robot_joints(start_joint_angles, speed_percentage=50)

            delay_robot(0.5)

        result = move_robot_joints(end_joint_angles, speed_percentage=10, wait_for_ack=True)

        delay_robot(1.0)
        
        # Return to standard position after search completion
        move_to_standard_position()

        return result
    
    elif search_pattern == "vertical":

        base_angle = 90

        if continue_from_current_position:
            current_joint_angles = get_robot_joint_angles()
            base_angle = current_joint_angles[0]

        # Vertical search pattern
        move_robot_joints([base_angle, -60, 180, 0, -30, 180], speed_percentage=30)
        delay_robot(0.5)
        move_robot_joints([base_angle, -80, 200, 0, -30, 180], speed_percentage=5)
        delay_robot(0.5)
        move_robot_joints([base_angle, -100, 220, 0, -60, 180], speed_percentage=5)

    elif search_pattern == "spiral":
        # Spiral search pattern
        move_to_standard_position()

    else:
        return "Unknown search pattern"
    


    #move_robot_joints([-30, -60, 180, 0, -45, 90], speed_percentage=50)

    #move_robot_joints([120, -60, 180, 0, -45, 90], speed_percentage=10)

def move_to_standard_position():
    result =  move_robot_joints([89.9384765625, -93.9459375, 197.5112171052632, 0.0, -81.39375, 180.039375], speed_percentage=40, wait_for_ack=True)
    return result



if __name__ == "__main__":

    #search(search_pattern = "spiral", position = "high", continue_from_current_position = False)

    #move_robot_pose([60.23478186002641255, 235.6484543489161, 150.84535215031866, 90, 180, 0], speed_percentage=30)

    #smooth_helix([60.23478186002641255, 220.6484543489161, 150.84535215031866], radius=50, height=100, pitch=20, speed_percentage=40)

    start_pose = [60.23478186002641255, 235.6484543489161, 150.84535215031866, 90, 180, 0]

    second_pose = [60.23478186002641255, 235.6484543489161, 175.84535215031866, 90, 180, 0]

    third_pose = [60.23478186002641255, 235.6484543489161, 200.84535215031866, 90, 180, 0]

    smooth_circle(start_pose[0:3], radius=50, speed_percentage=40, start_pose=start_pose)

    smooth_circle(second_pose[0:3], radius=50, speed_percentage=40, start_pose=second_pose)

    smooth_circle(third_pose[0:3], radius=50, speed_percentage=40, start_pose=third_pose)

    #move_to_standard_position()

    #print(get_robot_pose())

    #print(get_robot_joint_angles())

    #search()
    
    #search()

    #time.sleep(4.0)
    
    #stop_robot_movement()

    #wave_command()

    #move_robot_joints(joint_angles=[90.0, -41.313, 151.032, -9.731, -9.7, 180.0], duration=3.0)

    #home_robot()
