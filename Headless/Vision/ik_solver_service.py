# ik_and_manip_service.py
import numpy as np
import socket
from spatialmath import SE3
from spatialmath.base import trinterp
from collections import namedtuple
import time

# Import the robot instance from your file
from Headless.PAROL6_ROBOT import robot as robot_model

# --- NEW (Corrected Fix): Set joint limits on each individual link ---
# The robot's main .qlim property is read-only, so we must modify the
# qlim property of each link in the robot model's .links list.

joint_limits_deg = [
    [-123.046875, 123.046875], 
    [-145.0088, -3.375], 
    [107.866, 287.8675], 
    [-105.46975, 105.46975], 
    [-90, 90], 
    [0, 360]
]

# Convert the entire list to radians first
joint_limits_rad = np.deg2rad(joint_limits_deg)

# Iterate through the robot's links and assign the limits one by one
for i, link in enumerate(robot_model.links):
    link.qlim = joint_limits_rad[i]
# -------------------------------------------------------------------

IKResult = namedtuple('IKResult', 'success q iterations residual tolerance_used violations')

def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def unwrap_angles(q_solution, q_current):
    """Unwrap angles in the solution to be closest to current position."""
    q_unwrapped = q_solution.copy()
    for i in range(len(q_solution)):
        diff = q_solution[i] - q_current[i]
        if diff > np.pi:
            q_unwrapped[i] = q_solution[i] - 2 * np.pi
        elif diff < -np.pi:
            q_unwrapped[i] = q_solution[i] + 2 * np.pi
    return q_unwrapped

def calculate_adaptive_tolerance(robot, q, strict_tol=1e-10, loose_tol=1e-7):
    """Calculate adaptive tolerance based on proximity to singularities."""
    manip = robot.manipulability(np.array(q, dtype=float))
    singularity_threshold = 0.001
    sing_normalized = np.clip(manip / singularity_threshold, 0.0, 1.0)
    return loose_tol + (strict_tol - loose_tol) * sing_normalized

def calculate_configuration_dependent_max_reach(q_seed):
    """Calculate maximum reach based on joint configuration."""
    # ==========================================================
    # === FIX: Use the more conservative and accurate reach value ===
    # ==========================================================
    # The previous value of 0.5 was too optimistic. The robot's model in
    # headless_commander.py uses 0.44, which is a more realistic limit.
    # We will use this value to ensure consistency.
    base_max_reach = 0.44
    # ==========================================================
    
    j5_angle = q_seed[4] if len(q_seed) > 4 else 0.0
    j5_normalized = normalize_angle(j5_angle)
    dist_from_90_deg = min(abs(j5_normalized - np.pi / 2), abs(j5_normalized + np.pi / 2))
    reduction_range = np.pi / 4
    if dist_from_90_deg <= reduction_range:
        proximity_factor = 1.0 - (dist_from_90_deg / reduction_range)
        reach_reduction = 0.045 * proximity_factor
        return base_max_reach - reach_reduction
    return base_max_reach

def solve_ik_with_adaptive_tol_subdivision(
        robot, target_pose: SE3, current_q, current_pose: SE3 | None = None,
        max_depth: int = 4, ilimit: int = 100, jogging: bool = False):
    """The advanced, recursive IK solver."""
    if current_pose is None:
        current_pose = robot.fkine(current_q)

    def _solve(Ta: SE3, Tb: SE3, q_seed, depth, tol):
        current_reach = np.linalg.norm(Tb.t)
        max_reach_threshold = calculate_configuration_dependent_max_reach(q_seed)
        if current_reach >= max_reach_threshold:
            # This is a critical failure reason that we need to return
            reason = f"Reach limit exceeded: {current_reach:.4f}m >= {max_reach_threshold:.4f}m"
            return [], False, 0, 0, reason # Add reason to the return tuple
        
        res = robot.ikine_LMS(Tb, q0=q_seed, ilimit=ilimit, tol=tol, wN=1e-12)
        if res.success:
            q_good = unwrap_angles(res.q, q_seed)
            return [q_good], True, res.iterations, res.residual, "Success"
        if depth >= max_depth:
            return [], False, res.iterations, res.residual, "Max subdivision depth reached"
        
        Tc = SE3(trinterp(Ta.A, Tb.A, 0.5))
        # Pass the reason through the recursive calls
        left_path, ok_L, it_L, r_L, reason_L = _solve(Ta, Tc, q_seed, depth + 1, tol)
        if not ok_L:
            return [], False, it_L, r_L, reason_L
        
        q_mid = left_path[-1]
        right_path, ok_R, it_R, r_R, reason_R = _solve(Tc, Tb, q_mid, depth + 1, tol)
        return left_path + right_path, ok_R, it_L + it_R, r_R, reason_R

    adaptive_tol = 1e-10 if jogging else calculate_adaptive_tolerance(robot, current_q)
    # Capture the failure reason from the recursive solver
    path, ok, its, resid, reason = _solve(current_pose, target_pose, current_q, 0, adaptive_tol)
    
    if ok and path:
        final_q = path[-1]
        # Check joint limits
        if np.all(final_q >= robot_model.qlim[0, :]) and np.all(final_q <= robot_model.qlim[1, :]):
             return IKResult(True, final_q, its, resid, adaptive_tol, None)
        else:
            violations = []
            for i in range(robot_model.n):
                if not (robot_model.qlim[0, i] <= final_q[i] <= robot_model.qlim[1, i]):
                    violations.append(f"J{i+1}: {np.rad2deg(final_q[i]):.2f} deg")
            # Return the specific violation
            return IKResult(False, None, its, resid, adaptive_tol, f"Joint limit violation: {', '.join(violations)}")

    # Return the detailed reason for failure
    return IKResult(False, None, its, resid, adaptive_tol, reason)

# --- Main Service Loop ---
# Fix for ik_solver_service.py - update the main service loop

def main():
    """Listens for IK requests and provides detailed feedback."""
    HOST = "127.0.0.1"
    PORT = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))
        print(f"Unified Solver Service listening on {HOST}:{PORT}")
        while True:
            data, addr = s.recvfrom(1024)
            if not data: continue

            try:
                message = data.decode('utf-8')
                
                if ';' in message:
                    # --- IK Solver Logic with Fixed Response ---
                    parts = message.split(';')
                    matrix_str, q0_str = parts[0], parts[1]

                    target_matrix = np.array([float(v) for v in matrix_str.split(',')]).reshape((4, 4))
                    current_q_deg = np.array([float(v) for v in q0_str.split(',')])
                    current_q_rad = np.deg2rad(current_q_deg)
                    
                    # Debug log
                    timestamp = time.strftime('%H:%M:%S')
                    target_pos_str = np.array2string(target_matrix[:3, 3], precision=3, suppress_small=True)
                    print(f"[{timestamp}] Received IK Request for position: {target_pos_str}")

                    solution = solve_ik_with_adaptive_tol_subdivision(
                        robot=robot_model, target_pose=SE3(target_matrix), current_q=current_q_rad
                    )

                    if solution.success:
                        solution_q_deg = np.rad2deg(solution.q)
                        response = ",".join(map(str, solution_q_deg))
                        print(f"  -> SUCCESS: Solution found.")
                    else:
                        # CRITICAL FIX: Always send "FAIL" for failed IK
                        # The detailed reason is for logging only
                        response = "FAIL"
                        print(f"  -> FAIL: {solution.violations or 'IK solution not found'}")
                    
                    s.sendto(response.encode('utf-8'), addr)

                else:
                    # Manipulability calculation
                    q_deg = np.array([float(v) for v in message.split(',')])
                    q_rad = np.deg2rad(q_deg)
                    manip = robot_model.manipulability(q_rad)
                    s.sendto(str(manip).encode('utf-8'), addr)

            except Exception as e:
                print(f"An error occurred in unified service: {e}")
                # Send ERROR for any exceptions
                s.sendto(b"ERROR", addr)

if __name__ == "__main__":
    main()