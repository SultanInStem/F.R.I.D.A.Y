import sys
import os
import time
import socket
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commands.main import mc          # reuses the existing MyCobot320 handle
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 65432
 
MOVE_SPEED   = 10      # send_angles speed; keep <= 40 per existing guards
SETTLE_S     = 0.5     # extra dwell after is_moving() clears
GRIP_DWELL_S = 1.5     # time for the gripper to finish closing/opening
 
# Taught poses, all in DEGREES. Get these once with get_angles.py:
#   1. jog the arm by hand to the pose
#   2. run get_angles.py, paste the result here
HOME_ANGLES   = [0, 0, 0, 0, 0, 0]
LIFT_ANGLES   = [0, 20, 20, 0, 0, 0]      # <-- RETEACH: clear of the platform
BASKET_ANGLES = [0, 40, 0, 0, -90, 0]    # <-- RETEACH: over the basket
 
JOINT_LIMITS = [(-168, 168), (-135, 135), (-145, 145),
                (-148, 148), (-168, 168), (-175, 175)]

GRIPPER_OPEN_VALUE = 100
GRIPPER_CLOSED_VALUE = 10

# ─────────────────────────────────────────────
# MOTION HELPERS
# ─────────────────────────────────────────────
def wait_until_stopped(timeout=60.0):
    """Blocks until the arm reports it has stopped moving."""
    t0 = time.time()
    time.sleep(0.3)                     # is_moving() can lag the command
    while mc.is_moving():
        if time.time() - t0 > timeout:
            raise TimeoutError("arm still moving after timeout")
        time.sleep(0.1)
    time.sleep(SETTLE_S)
 
 
def move_angles(angles, speed=MOVE_SPEED, label=""):
    """Sends 6 joint angles in degrees and blocks until the motion completes."""
    if len(angles) != 6:
        raise ValueError("need exactly 6 joint angles")
    for i, (a, (lo, hi)) in enumerate(zip(angles, JOINT_LIMITS)):
        if not lo <= a <= hi:
            raise ValueError(f"J{i+1}={a:.1f} outside limit [{lo}, {hi}]")
    print(f"  -> {label or 'move'}: {[round(a, 1) for a in angles]}")
    mc.send_angles(list(angles), speed)
    wait_until_stopped()
 
 
def set_gripper(value, speed):
    """value 0 = fully closed, 100 = fully open."""
    if not 0 <= value <= 100:
        raise ValueError("gripper value must be 0-100")
    if not 0 < speed <= 100:
        raise ValueError("gripper speed must be 1-100")
    mc.set_gripper_mode(1)
    time.sleep(0.1)
    mc.set_gripper_value(int(value), int(speed))
    time.sleep(GRIP_DWELL_S)
 
 
# ─────────────────────────────────────────────
# THE PICK CYCLE
# ─────────────────────────────────────────────
def run_pick(joint_deg, grip_value, grip_speed):
    """
    target -> close -> lift -> basket -> release.
    Returns when the fruit has been released. Homing happens afterwards,
    outside the timed window.
    """
    set_gripper(GRIPPER_OPEN_VALUE, grip_speed)                       # open before approach
    move_angles(joint_deg,      label="target")
    set_gripper(GRIPPER_CLOSED_VALUE, grip_speed)                # close on the fruit
    move_angles(LIFT_ANGLES,    label="lift")
    move_angles(BASKET_ANGLES,  label="basket")
    set_gripper(GRIPPER_OPEN_VALUE, grip_speed)                       # release
    return True
 
 
def go_home():
    """Return to home. Deliberately does NOT recalibrate the gripper -
    that would change the closure reference partway through the run."""
    try:
        move_angles(HOME_ANGLES, label="home")
    except Exception as e:
        print(f"  ! homing failed: {e}")
 
 
# ─────────────────────────────────────────────
# COMMAND HANDLING
# ─────────────────────────────────────────────
def handle(line):
    """Returns (reply_string, should_home_afterwards)."""
    parts = [p.strip() for p in line.strip().split(",")]
    cmd = parts[0].upper()
 
    if cmd == "PING":
        return "PONG", False
 
    if cmd == "HOME":
        go_home()
        return "DONE", False
 
    if cmd == "PICK":
        if len(parts) != 9:
            return "FAIL,BAD_FORMAT", False
        try:
            joint_deg = [float(v) for v in parts[1:7]]
            grip_value = int(float(parts[7]))
            grip_speed = int(float(parts[8]))
        except ValueError:
            return "FAIL,BAD_NUMBERS", False
 
        try:
            run_pick(joint_deg, grip_value, grip_speed)
            return "DONE", True
        except ValueError as e:
            return f"FAIL,LIMIT:{e}", True
        except TimeoutError as e:
            return f"FAIL,TIMEOUT:{e}", True
        except Exception as e:
            return f"FAIL,ERROR:{type(e).__name__}:{e}", True
 
    return "FAIL,UNKNOWN_CMD", False
 
 
# ─────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────
def main():
    print("Homing before first trial...")
    go_home()
    set_gripper(100, 40)
 
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"Pick server listening on {HOST}:{PORT}")
 
    try:
        while True:
            conn, addr = srv.accept()
            with conn:
                conn.settimeout(120.0)
                try:
                    data = conn.recv(1024).decode()
                except socket.timeout:
                    print(f"{addr} timed out before sending")
                    continue
                if not data:
                    continue
 
                line = data.splitlines()[0]
                print(f"\n[{time.strftime('%H:%M:%S')}] {addr[0]} -> {line}")
 
                reply, should_home = handle(line)
 
                try:
                    conn.sendall((reply + "\n").encode())
                except socket.error as e:
                    print(f"  ! could not ack: {e}")
                print(f"  <- {reply}")
 
            # after the ack and after the socket closes, outside the timed window
            if should_home:
                go_home()
                print("  ready for next trial")
 
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        srv.close()
        go_home()
 
 
if __name__ == "__main__":
    main()
