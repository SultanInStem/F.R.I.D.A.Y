# F.R.I.D.A.Y. — AI-Powered Strawberry Harvester

Autonomous ripeness-selective strawberry harvesting with a YOLOv8m detector and a
six-DOF myCobot 320 Pi manipulator.

This repository is the software artifact for the paper *"AI-Powered Strawberry
Harvester Using a Robotic Arm"* (H. Kulhandjian, N. Khudayberdiev, Department of
Electrical and Computer Engineering, California State University, Fresno).
Supported in part by EDA Project Grant 077907908, Fresno-Merced Future of Food
Innovation (F3) Coalition.

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Demo](#2-demo)
3. [Installation](#3-installation)
4. [Coordinate frames and conventions](#4-coordinate-frames-and-conventions)
5. [Running the system](#5-running-the-system)
6. [Critical code walkthrough](#6-critical-code-walkthrough)
7. [Configuration reference](#7-configuration-reference)
8. [Experiment logging format](#8-experiment-logging-format)
9. [Known issues and open items](#9-known-issues-and-open-items)
10. [Legacy and deprecated files](#10-legacy-and-deprecated-files)

---

## 1. System overview

The system splits across two machines connected over a private Ethernet link.
Perception and motion planning run on the Jetson; only low-level joint commands
cross the wire.

```
┌─────────────────────────────── NVIDIA Jetson AGX Orin ───────────────────────────────┐
│                                                                                      │
│   RealSense D435 ──► align(depth→color) ──► YOLOv8 ONNX (cv2.dnn) ──► bounding box    │
│                              │                                              │        │
│                              ▼                                              ▼        │
│                    median 5×5 depth patch ────────────► rs2_deproject_pixel_to_point  │
│                                                                    │                 │
│                                                        point in CAMERA frame (m)     │
│                                                                    │                 │
│                                              T_cam2base (cam2base.json, 4×4)         │
│                                                                    ▼                 │
│                                                        point in BASE frame (m)       │
│                                                                    │                 │
│                                      + GRIPPER_LENGTH on Z (flange, not fingertip)   │
│                                                                    │                 │
│                                    ikpy inverse_kinematics, position-only            │
│                                                                    ▼                 │
│                                                        6 joint angles (degrees)      │
└────────────────────────────────────────────────────────────┬─────────────────────────┘
                                                             │
                       TCP :65432   "PICK,j1..j6,grip,speed\n"│
                                                             ▼
┌─────────────────────────── Raspberry Pi 4B (inside myCobot 320) ─────────────────────┐
│                                                                                      │
│   pick_server.py ──► joint-limit check ──► pymycobot send_angles() ──► servos         │
│                                        └─► gripper open/close ──► "DONE\n" / "FAIL,…" │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

Why the split: the onboard Raspberry Pi 4B cannot run YOLOv8m inference at a
useful frame rate, and the Jetson has no direct serial line to the servos. The
Pi owns the hardware; the Jetson owns the intelligence. The socket is the only
coupling between them, which also means either side can be tested in isolation.

---

## 2. Demo

Drop media into a `media/` folder at the repository root and reference it with a
relative path. GitHub renders GIFs inline; keep each under ~10 MB.

### Full pick cycle

<p align="center">
  <img src="media/pick_cycle.gif" width="640" alt="Full autonomous pick cycle">
</p>

*Caption: detection lock → approach → grasp → lift → basket release.*

### Detection

<p align="center">
  <img src="media/detection_ripe.png" width="310" alt="Ripe detection">
  <img src="media/detection_mixed.png" width="310" alt="Mixed-ripeness detection">
</p>

*Caption: live YOLOv8m output with ripe / unripe / rotten labels.*

### Hardware setup

<p align="center">
  <img src="media/setup.jpg" width="640" alt="Rig overview: myCobot 320 Pi, RealSense D435, Jetson AGX Orin">
</p>

*Caption: myCobot 320 Pi, overhead RealSense D435, Jetson AGX Orin.*

### Calibration

<p align="center">
  <img src="media/calibration.gif" width="640" alt="Automatic ChArUco hand-eye calibration">
</p>

*Caption: `calibrate.py` stepping through calibration poses.*

For a video longer than a few seconds, upload it to YouTube and link a thumbnail:

```markdown
[![Demo video](media/video_thumbnail.jpg)](https://youtu.be/VIDEO_ID)
```

---

## 3. Installation

### 3.1 Jetson (perception side)

PyTorch and torchvision must be installed separately from the NVIDIA-provided
wheels so that CUDA is available — do not let pip pull the generic builds.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas tqdm PyYAML pillow requests
pip install ultralytics ikpy==3.4.2 pymycobot==3.6.1
pip install pyrealsense2
pip install opencv-contrib-python==4.10.0.84   # see note below
```

**The OpenCV pin is not optional.** OpenCV 5.x removed `cv2.calibrateHandEye`
and `cv2.aruco.interpolateCornersCharuco` from the Python bindings, both of
which `calibrate.py` depends on. The script fails fast rather than
crashing halfway through a capture session:

```python
# calibrate.py (top of file)
if not hasattr(cv2, "calibrateHandEye"):
    raise SystemExit(
        f"OpenCV {cv2.__version__} does not expose calibrateHandEye.\n"
        "Install a 4.x build:  pip install opencv-contrib-python==4.10.0.84"
    )
```

`pyrealsense2` also has to match the librealsense version flashed on the device;
a mismatch shows up as `wait_for_frames()` timing out rather than as an import
error, which is a slow way to find out.

### 3.2 Raspberry Pi (arm side)

```bash
pip install pymycobot==3.6.1 ikpy==3.4.2 numpy
```

No OpenCV, no torch — the Pi never sees an image.

### 3.3 Network

The Jetson and Pi sit on a static private subnet. Defaults in the code:

| Role   | Address        | Port  |
|--------|----------------|-------|
| Jetson | `192.168.10.1` | —     |
| Pi     | `192.168.10.2` | 65432 |

### 3.4 Credentials

Host, username, and connection parameters live in
`arm_controls/utils/globals.py`. **Do not commit secrets to this file** — it is
tracked in a public repository. Read them from the environment instead:

```python
import os
NVIDIA_HOST     = os.environ.get("NVIDIA_HOST", "192.168.10.1")
NVIDIA_USER     = os.environ.get("NVIDIA_USER", "usr2")
NVIDIA_PASSWORD = os.environ["NVIDIA_PASSWORD"]   # no default; fail loudly
```

A credential was previously committed here. It must be rotated on the device
and purged from history with `git filter-repo`; deleting the line in a new
commit does not remove it from the object store.

---

## 4. Coordinate frames and conventions

Four frames are in play, and most integration bugs in this project have come
from confusing two of them.

| Frame | Origin | Units | Notes |
|-------|--------|-------|-------|
| **Camera** | D435 colour imager optical centre | metres | `+Z` forward along the optical axis, `+X` right, `+Y` down |
| **Base** (`F0`) | Top of the grey base housing, *above* the table surface | metres | Objects resting on the table have **negative Z** |
| **Flange** | Joint-6 output face | metres | What ikpy's forward kinematics returns |
| **Tip** | Gripper fingertip / pointer bolt | metres | Flange + `GRIPPER_LENGTH` along the approach vector |

Three conventions that are easy to get wrong:

**Angles.** `pymycobot` speaks degrees. `ikpy` speaks radians. Every boundary
between them converts explicitly. ikpy's angle vector has **8 entries** (a fixed
base link, 6 revolute joints, a fixed gripper link) — the 6 real joints are
`angles[1:7]`.

**Intrinsics.** After `rs.align(rs.stream.color)`, the depth frame has been
resampled into the colour imager's geometry. Deprojecting with the *depth
stream's* intrinsics after aligning produces a systematic scale error, because
the two imagers have different fields of view. Intrinsics must be pulled from
the aligned frame, inside the loop:

```python
# autonomous_test.py:260
intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
```

**The camera→base transform is unconditional.** It is a single 4×4 homogeneous
matrix applied by matrix multiply. It is *not* a set of per-axis offsets with
sign flips chosen by quadrant — that approach appears in the legacy scripts
(§10) and is wrong; it cannot represent the rotation between the frames at all.

---

## 5. Running the system

### 5.1 Calibrate (once, and again after anything moves)

```bash
cd vision_controls

python calibrate.py board      # writes charuco_board.png
# print at 300 dpi with scaling OFF, measure a square with calipers,
# update SQUARE_LENGTH_M / MARKER_LENGTH_M at the top of the file

python calibrate.py            # AUTO: move arm, capture, solve, write cam2base.json
python calibrate.py solve      # re-solve from an existing handeye_poses.json
```

Before running AUTO mode, leave the arm in a pose where the camera can see the
board, or set `HOME_ANGLES` at the top of the file. The script then:

1. Generates `N_POSES` joint configurations by perturbing each joint around home
   (`PERTURB_DEG`), clipped to the joint limits with a 5° margin.
2. Moves to each pose, waits `SETTLE_S` for vibration to die out, and keeps the
   best of `FRAMES_PER_POSE` board detections (PnP reprojection ≤ `MAX_REPROJ_PX`).
3. Returns home and solves with **every** OpenCV hand-eye method, keeping the one
   with the lowest residual.
4. Drops outlier poses (residual > max(3 × median, 3 mm)) and re-solves once.
5. Writes `cam2base.json` with the transform, RMS residual, pose count, chosen
   method, and dropped pose indices.

Perturbations are small on the shoulder and elbow (to stay clear of the table
and boom) and large on the wrist, because hand-eye calibration cannot observe
the rotational component from pure translation. The solver warns when the RMS
residual exceeds 5 mm.

The residual is the noise floor for everything downstream — no amount of IK
tuning produces end-effector accuracy better than the calibration that positions
the camera.

### 5.2 Start the arm server (on the Pi)

```bash
cd arm_controls/commands
python pick_server.py
```

It homes the arm, opens the gripper, and listens on `0.0.0.0:65432`.

### 5.3 Run the autonomous trials (on the Jetson)

```bash
cd vision_controls
python autonomous_test.py
```

Place one fruit in view. The window shows the live detection and a stability
counter. When the target holds still for `DETECTION_FRAMES` consecutive frames
the trial fires automatically, the cycle timer starts, and after the arm acks
you are prompted at the terminal for the outcome and failure mode. Every trial
is appended to `experiment_b_log.csv` and fsync'd immediately, so a crash mid-run
costs at most the current trial.

Press `q` to stop.

---

## 6. Critical code walkthrough

### 6.1 The kinematic chain

```python
# autonomous_test.py:16-22
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ikpy.chain import Chain
    chain = Chain.from_urdf_file(
        "./mycobot_320pi.urdf",
        active_links_mask=[False, True, True, True, True, True, True, False]
    )
```

The URDF describes 8 links: `base_link`, `link1`…`link6`, and `gripper`. The
mask marks the first and last as **fixed**. Without this argument, ikpy treats
the base and the gripper as solvable degrees of freedom and returns solutions
for an 8-DOF arm that the hardware cannot execute — the first and last entries
come back non-zero and get silently discarded downstream, so the flange lands
somewhere other than the commanded point.

The mask must be identical everywhere a chain is constructed. It currently is
not (see §9).

### 6.2 Inverse kinematics with a residual gate

```python
def compute_angles(point_in_base_frame):
    """Returns the full ikpy angle vector (radians), or [] if unreachable."""
    angles = chain.inverse_kinematics(
        point_in_base_frame,
        optimizer="least_squares",
        max_iter=1000,
    )
    achieved = chain.forward_kinematics(angles)[:3, 3]
    residual = np.linalg.norm(achieved - point_in_base_frame)
    if residual > 0.005:
        print(f"  IK residual {residual*1000:.1f} mm - point outside working radius")
        return []
    return angles
```

Two decisions are encoded here.

**Position-only IK.** No target orientation is passed, so the solver is free to
use all six joints to reach the point. An earlier version constrained the
flange's Z axis to point straight down (`orientation_mode="Z"`, `[0, 0, -1]`).
That constraint removed orientation-dependent failures but cost reach: every
IK failure in the constrained trials clustered beyond ~331 mm from the base.
Dropping it enlarges the reachable workspace, at the cost of a less predictable
approach direction — the gripper may arrive tilted rather than vertical.

**The residual check is the reachability test.** ikpy's `inverse_kinematics` is
a numerical optimiser — it always returns *something*, and for an unreachable
target that something is the closest achievable pose, silently. Round-tripping
the solution through forward kinematics and measuring how far the result landed
from the request is the only reliable way to detect this. A naive radius check
(`‖p‖ < reach`) is not equivalent: it accepts points that are inside the sphere
but outside the actual workspace, such as points blocked by joint limits or
inside the base column.

**5 mm is the gate.** Below the ~7 mm calibration residual, so it never rejects
a target the rest of the system could have hit anyway.

```python
# autonomous_test.py:82-84
def angles_to_degrees(angles):
    """ikpy vector (8 links) -> the 6 joint angles in degrees for pymycobot."""
    return [round(float(np.degrees(a)), 3) for a in angles[1:7]]
```

The `[1:7]` slice drops the two fixed links. Rounding to 3 decimal places in
degrees is ~0.001° — far below the servo resolution, so it costs nothing.

### 6.3 Depth sampling

```python
# autonomous_test.py:216-224
def median_depth(depth_frame, cx, cy, half=DEPTH_PATCH // 2):
    """Median of the valid depths in a small window - robust to specular dropouts."""
    vals = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            d = depth_frame.get_distance(cx + dx, cy + dy)
            if d > 0:
                vals.append(d)
    return float(np.median(vals)) if vals else 0.0
```

Sampling a single pixel at the bounding-box centre fails on fruit. Strawberries
are glossy, and a specular highlight causes the IR stereo matcher to drop out at
exactly the pixel you care about — `get_distance` returns `0.0`, which is
indistinguishable from "the object is at the camera". A 5×5 median over valid
returns tolerates a scattering of dropouts and rejects outliers rather than
averaging them in, which a mean would not.

Two related constraints on the sensor: the D435 is unreliable below roughly
0.3 m, and `d > 0` is the validity test, not a plausibility test — zeros are
excluded, but a wildly wrong non-zero value still gets through and is caught, if
at all, by the stability buffer in §6.5.

### 6.4 Camera to base, and the flange offset

```python
# autonomous_test.py:286-288
point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
p_base = (T_CAM2BASE @ np.array([*point_3d, 1.0]))[:3]
p_base = [p_base[0], p_base[1], p_base[2] + GRIPPER_LENGTH]
```

Line by line: the pixel plus its depth becomes a metric 3-D point in the camera
frame; the homogeneous multiply by the calibrated 4×4 moves it into the base
frame (note the `1.0` appended to make it homogeneous — with a `0.0` it would be
treated as a direction and the translation would be dropped); and the last line
adds the flange-to-tip distance.

That last line deserves care. ikpy solves for the **flange**, but the fruit is
where the **fingertips** need to be. Adding a constant `+GRIPPER_LENGTH` on Z
commands the flange to sit 13 cm above the fruit, which puts the tip on it only
when the gripper points straight down. With position-only IK (§6.2) the approach
direction is no longer fixed, so for tilted solutions the tip lands off-target
by up to `GRIPPER_LENGTH · sin(tilt)`. The exact fix is to make the tip the end
of the kinematic chain — set the `wrist_to_gripper` joint origin in the URDF to
the measured flange-to-tip distance — and solve for the tip position directly.

`GRIPPER_LENGTH = 0.13` (`autonomous_test.py:27`) is a measured quantity, not a
datasheet value, and must be re-measured if the end-effector is changed.

### 6.5 Detection stability buffer

```python
# autonomous_test.py:299-309
            if best is None or conf > best[1]:
                best = (label, conf, p_base, centered)

        in_cooldown = (time.time() - armed_at) < COOLDOWN_S
        if best is not None and best[3] and not in_cooldown:
            coords_buffer.append(best[2])
            conf_buffer.append(best[1])
        else:
            coords_buffer.clear()      # any dropped frame restarts the count
            conf_buffer.clear()
```

```python
# autonomous_test.py:319-324
        if len(coords_buffer) >= DETECTION_FRAMES:
            arr = np.array(coords_buffer)
            avg = arr.mean(axis=0)
            sd = arr.std(axis=0)

            if np.all(sd < STABILITY_TOL):
```

This is the trigger, and it does three jobs at once.

It **prevents firing on a transient.** A single frame of bad depth or a
half-occluded box produces a wild coordinate; requiring 15 consecutive frames
means such a frame resets the count to zero rather than launching the arm.

It **averages down sensor noise.** The mean of 15 samples reduces the
zero-mean component of the depth noise by roughly √15.

It **verifies its own precondition.** Averaging is only valid if the samples
describe the same static target. `np.all(sd < STABILITY_TOL)` checks the
per-axis standard deviation across the buffer against 10 mm before the mean is
trusted. If the fruit or the camera was moving, the SD blows up, the buffer is
discarded, and nothing is sent to the arm.

`COOLDOWN_S` blocks re-arming for 3 seconds after a trial so the system does not
immediately re-detect the same fruit while the arm is still clearing the frame.

**Note:** `CENTER_THRESHOLD = 9999` (line 55) makes the `centered` flag
unconditionally true, disabling the centring gate. See §9.

### 6.6 The Jetson→Pi protocol

```python
# autonomous_test.py:90-109
def send_to_pi(joint_deg, gripper_value, gripper_speed):
    """
    Wire format out : "PICK,j1,j2,j3,j4,j5,j6,grip_value,grip_speed\n"
    Wire format back: "DONE\n" or "FAIL,<reason>\n"
    """
    msg = "PICK," + ",".join(f"{a:.3f}" for a in joint_deg)
    msg += f",{gripper_value},{gripper_speed}\n"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(PI_TIMEOUT)
            s.connect((PI_HOST, PI_PORT))
            s.sendall(msg.encode())
            reply = s.recv(1024).decode().strip()
        return reply.startswith("DONE"), reply
    except socket.timeout:
        return False, "TIMEOUT"
    except socket.error as e:
        return False, f"SOCKET_ERROR:{e}"
```

One connection per trial, synchronous, plain ASCII. The design is deliberately
minimal: the protocol is debuggable with `netcat`, a hung arm surfaces as a
`TIMEOUT` row in the log rather than a hung experiment, and a network fault is
recorded as a distinct failure mode instead of being scored as a picking
failure. Nothing is retried — a retry would corrupt the cycle-time measurement.

Supported commands on the server side (`pick_server.py:100-132`):

| Command | Arguments | Reply |
|---------|-----------|-------|
| `PING` | — | `PONG` |
| `HOME` | — | `DONE` |
| `PICK` | 6 joint angles (deg), gripper value, gripper speed | `DONE` / `FAIL,<reason>` |

Failure replies distinguish `BAD_FORMAT`, `BAD_NUMBERS`, `LIMIT:…`,
`TIMEOUT:…`, and `ERROR:<type>:<msg>`, all of which land in the `pi_reply`
column of the CSV.

### 6.7 The pick cycle

```python
# pick_server.py:73-85
def run_pick(joint_deg, grip_value, grip_speed):
    """
    target -> close -> lift -> basket -> release.
    Returns when the fruit has been released. Homing happens afterwards,
    outside the timed window.
    """
    set_gripper(GRIPPER_OPEN_VALUE, grip_speed)      # open before approach
    move_angles(joint_deg,      label="target")
    set_gripper(GRIPPER_CLOSED_VALUE, grip_speed)    # close on the fruit
    move_angles(LIFT_ANGLES,    label="lift")
    move_angles(BASKET_ANGLES,  label="basket")
    set_gripper(GRIPPER_OPEN_VALUE, grip_speed)      # release
    return True
```

The gripper opens *before* the approach, not after arrival — a closed gripper
travelling to the target sweeps neighbouring fruit off the plant.

Homing is deliberately excluded from `run_pick` and performed by the server
after the ack has been sent (`pick_server.py:174-176`), so the return-to-home
travel does not inflate the reported cycle time. `go_home()` also does *not*
recalibrate the gripper, which would reset the closure reference partway through
a run and make trials incomparable:

```python
# pick_server.py:88-90
def go_home():
    """Return to home. Deliberately does NOT recalibrate the gripper -
    that would change the closure reference partway through the run."""
```

Every commanded pose is bounds-checked against the manufacturer's joint limits
before it reaches a servo:

```python
# pick_server.py:26-27
JOINT_LIMITS = [(-168, 168), (-135, 135), (-145, 145),
                (-148, 148), (-168, 168), (-175, 175)]

# pick_server.py:50-52
    for i, (a, (lo, hi)) in enumerate(zip(angles, JOINT_LIMITS)):
        if not lo <= a <= hi:
            raise ValueError(f"J{i+1}={a:.1f} outside limit [{lo}, {hi}]")
```

This is the last line of defence. An IK solution that satisfies the residual
check can still require a joint angle the hardware cannot reach; without this
guard the arm drives into a hard stop. The raised `ValueError` becomes a
`FAIL,LIMIT:` reply, which the Jetson records rather than treating as a
mechanical failure.

Blocking on motion completion is done by polling, with a lead-in delay because
`is_moving()` lags the command it is reporting on:

```python
# pick_server.py:35-43
def wait_until_stopped(timeout=60.0):
    t0 = time.time()
    time.sleep(0.3)                     # is_moving() can lag the command
    while mc.is_moving():
        if time.time() - t0 > timeout:
            raise TimeoutError("arm still moving after timeout")
        time.sleep(0.1)
    time.sleep(SETTLE_S)
```

Without the 0.3 s lead-in, `is_moving()` can still report `False` from before
the command was received and the function returns immediately, so the next
motion is issued into a still-moving arm. `SETTLE_S` allows mechanical
oscillation to damp out before the gripper closes.

### 6.8 Hand-eye calibration

This is an **eye-to-hand** configuration: the camera is fixed to the overhead
boom and the ChArUco board is mounted to the moving flange. OpenCV's
`calibrateHandEye` is written for the eye-*in*-hand case (camera on the flange),
so the robot poses must be inverted before being passed in:

```python
# calibrate.py — _solve_once()
    # eye-to-hand: feed the INVERSE of the flange pose
    R_b2g = [T[:3, :3].T for T in T_g2b]
    t_b2g = [(-T[:3, :3].T @ T[:3, 3]).reshape(3, 1) for T in T_g2b]
    R_t2c = [T[:3, :3] for T in T_t2c]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in T_t2c]

    R, t = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=method)
```

The inverse of a rigid transform is `R⁻¹ = Rᵀ`, `t⁻¹ = −Rᵀt` — that is exactly
what the first two lines construct. Skipping this inversion is the single most
common way to get a confident, completely wrong transform out of this function;
it typically produces a camera position mirrored through the base.

The residual is then computed independently of the solver, which matters because
`calibrateHandEye` reports no error metric of its own:

```python
# calibrate.py — _solve_once()
    pts = np.array([(np.linalg.inv(Tg) @ T_c2b @ Tt)[:3, 3]
                    for Tg, Tt in zip(T_g2b, T_t2c)])
    centroid = pts.mean(axis=0)
    resid = np.linalg.norm(pts - centroid, axis=1)
```

The board is bolted to the flange, so its position *in the flange frame* is a
fixed physical constant — the same in every sample. Chaining the solved
transform back through each sample should therefore recover the same point every
time, and the scatter of those recovered points is a direct, unbiased measure of
the calibration error.

`_best_method()` runs this for every OpenCV hand-eye method and keeps the lowest
RMS; `solve()` then drops poses whose residual exceeds max(3 × median, 3 mm) and
re-solves once, provided at least `MIN_SAMPLES` poses remain.

### 6.9 Detection preprocessing

```python
# autonomous_test.py:171-180
def letterbox(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (tw - new_w) // 2, (th - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y
```

The camera streams 640×480; the network expects 640×640. Stretching to square
would distort aspect ratio and shift every box centre, which propagates directly
into a 3-D localisation error. Letterboxing preserves the aspect ratio and pads
with grey (114, the Ultralytics convention, matching what the model saw during
training). The `scale`, `pad_x`, `pad_y` returned are then used to invert the
transform on the detections:

```python
# autonomous_test.py:198-200
        cx = (det[0] - pad_x) / scale
        cy = (det[1] - pad_y) / scale
        bw, bh = det[2] / scale, det[3] / scale
```

Subtract the padding first, then divide by the scale — reversing that order puts
every box off by the pad offset.

```python
# autonomous_test.py:189
    output = np.squeeze(outputs[0]).T
```

YOLOv8's ONNX head emits `[1, 4 + n_classes, 8400]`. The transpose turns it into
`[8400, 4 + n_classes]` so each row is one candidate. Unlike YOLOv5 there is no
separate objectness score, so the class scores start at index 4 — hence
`scores = det[4:]` on line 193, with the class confidence taken directly as the
detection confidence.

---

## 7. Configuration reference

### `vision_controls/autonomous_test.py`

| Constant | Value | Meaning |
|----------|-------|---------|
| `MODEL_PATH` | *(see §9)* | ONNX weights loaded by `cv2.dnn` |
| `NAMES_PATH` | *(see §9)* | class-name list, one per line, order must match training |
| `TARGET_CLASS` | *(see §9)* | only this class is picked; `None` picks any |
| `CONF_THRESH` | *(see §9)* | detection confidence floor; **must equal the value in Table I** |
| `NMS_THRESH` | 0.4 | IoU threshold for non-max suppression |
| `INPUT_SIZE` | (640, 640) | network input, matches training resolution |
| `DETECTION_FRAMES` | 15 | consecutive stable frames required to fire a trial |
| `STABILITY_TOL` | 0.010 m | max per-axis SD across the buffer |
| `CENTER_THRESHOLD` | 9999 px | centring gate — currently disabled, see §9 |
| `BRIGHTNESS_THRESHOLD` | 10 | mean ROI grey level below which a detection is dropped |
| `DEPTH_PATCH` | 5 | side length of the median depth window |
| `GRIPPER_LENGTH` | 0.13 m | measured flange-to-tip distance |
| `GRIPPER_VALUE` | 20 | closure command sent to the Pi (0 = closed, 100 = open) |
| `GRIPPER_SPEED` | 40 | gripper actuation speed |
| `COOLDOWN_S` | 3.0 s | dead time after a trial before re-arming |
| `PI_TIMEOUT` | 60.0 s | how long to wait for the Pi's ack |

### `arm_controls/commands/pick_server.py`

| Constant | Value | Meaning |
|----------|-------|---------|
| `MOVE_SPEED` | 10 | `send_angles` speed; low for repeatability |
| `SETTLE_S` | 0.5 s | dwell after `is_moving()` clears |
| `GRIP_DWELL_S` | 1.5 s | time allowed for the gripper to finish actuating |
| `HOME_ANGLES` | `[0,0,0,0,0,0]` | zero pose |
| `LIFT_ANGLES` | `[0,20,20,0,0,0]` | clear of the platform after grasping — **reteach per setup** |
| `BASKET_ANGLES` | `[0,-40,0,0,90,0]` | over the collection basket — **reteach per setup** |
| `GRIPPER_OPEN_VALUE` | 100 | fully open |
| `GRIPPER_CLOSED_VALUE` | 10 | closure used during a pick — see §9 |

`LIFT_ANGLES` and `BASKET_ANGLES` are taught poses, valid only for one physical
layout. Re-teach them with `scripts/get_angles.py` whenever the rig moves.

### `vision_controls/calibrate.py`

| Constant | Value | Meaning |
|----------|-------|---------|
| `SQUARES_X`, `SQUARES_Y` | 5, 7 | board dimensions in chessboard squares |
| `SQUARE_LENGTH_M` | 0.030 | **measure the printed board with calipers** |
| `MARKER_LENGTH_M` | 0.022 | must be strictly less than the square length |
| `ARUCO_DICT` | `DICT_5X5_100` | marker dictionary |
| `ROBOT_IP`, `ROBOT_PORT` | — | arm socket address used in AUTO mode |
| `HOME_ANGLES` | `None` | centre pose; `None` uses the arm's current pose |
| `N_POSES` | 20 | poses captured, including home |
| `PERTURB_DEG` | `[10, 8, 8, 20, 20, 30]` | max per-joint perturbation around home |
| `SPEED` | 10 | `send_angles` speed during capture |
| `SEED` | 0 | change for a different random pose set |
| `SETTLE_S` | 1.0 s | dwell after arrival before capturing |
| `FRAMES_PER_POSE` | 8 | best of N detections kept per pose |
| `MIN_CORNERS` | 10 | of 24 inner corners required for a valid detection |
| `MAX_REPROJ_PX` | 1.0 | PnP reprojection error ceiling |
| `MIN_SAMPLES` | 8 | minimum poses required to solve |

A wrong `SQUARE_LENGTH_M` scales the entire solved transform. It is the first
thing to check when the residual is large.

---

## 8. Experiment logging format

`autonomous_test.py` writes `experiment_b_log.csv` with one row per trial:

| Column | Source | Notes |
|--------|--------|-------|
| `trial` | auto-increment | resumes from existing row count on restart |
| `target_id` | derived | `T01`, `T02`, … |
| `timestamp` | wall clock | |
| `target_x_mm`, `target_y_mm`, `target_z_mm` | buffer mean | base frame, millimetres |
| `confidence` | buffer mean | mean detection confidence over the stable window |
| `pred_class` | detector | predicted class label |
| `detected` | constant `Y` | a row only exists if detection succeeded |
| `outcome` | operator | `S` or `F` |
| `failure_mode` | operator | `1`–`6`, see taxonomy below |
| `cycle_time_s` | `perf_counter` | detection-lock to Pi ack, excludes homing |
| `ik_ok` | solver | `N` auto-scores the trial as failure mode 5 |
| `pi_reply` | socket | raw reply string, including `FAIL,…` reasons |
| `notes` | operator | free text |

Failure taxonomy: **1** detection miss · **2** localisation error · **3** grasp
slip · **4** detachment failure · **5** IK failure · **6** collision.

```python
# autonomous_test.py:131-139
def append_row(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
```

The `flush` + `fsync` pair forces the row to physical storage before the next
trial begins. Buffered writes lose the tail of the run on a crash or an
unplanned power cut, which is exactly when you least want to lose data.

Note that a row is only written when the detector locked onto a target, so
failure mode 1 (detection miss) cannot be captured automatically — it has to be
recorded by the operator alongside the log.

---

## 9. Known issues and open items

These are tracked here rather than in issues so that anyone reading the paper's
artifact sees them.

**1. Model path and target class do not match the paper.** The runnable scripts
load `yolov8n_apples/my_model.onnx` with `TARGET_CLASS = "apple"`. The paper
reports a YOLOv8m three-class strawberry model. Point `MODEL_PATH` and
`NAMES_PATH` at `AI_model/yolov8m_strawberry/` and set `TARGET_CLASS = "ripe"`.
The strawberry `.names` file orders the classes `unripe, ripe, rotten`, so index
1 is the harvest target — the label string, not the index, is what
`TARGET_CLASS` is compared against, so the order only matters if the code is
changed to filter by index.

**2. `CONF_THRESH` is inconsistent across files.** `autonomous_test.py` uses
0.60; `test.py`, `manual_test.py`, and `AI_model/sub_main02.py` use 0.5; Table I
of the paper reports 0.70. One value must be chosen and propagated, and the
confusion matrix must be regenerated at the same threshold
(`model.val(conf=<T>, iou=0.5, plots=True, split="val")`) or the table and the
matrix will disagree.

**3. `run_pick` ignores its `grip_value` argument.** `pick_server.py:73` accepts
`grip_value` but line 81 closes to the module constant `GRIPPER_CLOSED_VALUE = 10`.
The Jetson sends 20. Whatever closure value is reported in the paper must be the
one actually executed.

**4. Two different chain constructions.** `arm_controls/utils/funcs.py:8` builds
the ikpy chain with no `active_links_mask`, while `autonomous_test.py` and
`calibrate.py` both use the masked form. Per §6.1, the unmasked chain
solves a different problem. `funcs.py` additionally checks reach against
`0.5**2` (a 500 mm radius for a 320 mm arm), performs that check *after* the
solve, and has no residual gate.

**5. `convert_to_radians` rounds to 2 decimal places.** `funcs.py:48`. 0.01 rad
is 0.57°; propagated through six joints in `compute_fk` this injects
millimetre-scale error into `get_arms_position()` for no benefit.

**6. `CENTER_THRESHOLD = 9999` disables the centring gate.** The `centered` flag
is unconditionally true. Either restore a meaningful pixel threshold or remove
the check so the code matches what is described.

**7. Serial connection opens at import time.** `commands/main.py:10` instantiates
`MyCobot320` at module scope, so any import of the module — including
`pick_server.py`'s `from commands.main import mc` — opens the port. Connection
failures then surface as import errors, and two processes cannot both import it.
Wrap in a `get_arm()` accessor.

**8. Mixed `sys.path` idioms in `scripts/`.** `calibrate_gripper.py`,
`calibrate_joints.py`, `diagnostic.py`, `get_angles.py`, `get_position.py`, and
`move_to_location.py` use `sys.path.append(os.path.abspath(".."))`, which
resolves against the working directory rather than the script location. The
other four already use the correct form:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

Related: `funcs.py:8` loads `"../mycobot_320pi.urdf"` and `main.py:32` writes
`"../../current_location.txt"`, both relative to the working directory.

**9. `requirements.txt` is incomplete.** It omits numpy, opencv, pyrealsense2,
ultralytics, and onnx, and carries unused entries inherited from the YOLOv5
requirements file (`gitpython`, `thop`, `seaborn`). Use §3 until it is rebuilt.

**10. Repository hygiene.** Roughly 330 MB of model weights are tracked,
including two near-identical 100 MB ONNX files, giving a ~274 MB `.git`.
`__pycache__` directories are committed despite `.gitignore`, including an
orphaned `compute_ik.cpython-312.pyc` with no corresponding source file.
`current_location.txt` is runtime state and should not be tracked. Consider Git
LFS or a release attachment for the weights.

---

## 10. Legacy and deprecated files

Kept for provenance; **not** part of the reported pipeline.

| Path | Status |
|------|--------|
| `vision_controls/test.py` | Earlier fork of `autonomous_test.py`, no CSV logging or socket protocol |
| `vision_controls/manual_test.py` | Older fork still using **sign-conditional per-axis offsets** (`x - CAMERA_X_OFFSET`, `-y + CAMERA_Y_OFFSET`, `CAMERA_Z_OFFSET - z`) and `gripper_length = 0.02`. This coordinate handling is superseded by the 4×4 transform in §4 and should not be reused. |
| `vision_controls/AI_model/sub_main01.py` | Ultralytics-based detection loop; writes coordinates to a text file and ships the file over a socket |
| `vision_controls/AI_model/sub_main02.py` | `cv2.dnn` port of `sub_main01.py`, same file-transfer scheme |
| `vision_controls/AI_model/send_coords.py` | Standalone socket file-sender test |
| `vision_controls/camera/test1.py` | RealSense colour + depth colormap viewer |
| `vision_controls/camera/test2.py` | Click-to-deproject viewer for spot-checking depth |
| `vision_controls/AI_model/yolov8m_apples/`, `yolov8n_apples/` | Earlier fruit models |

The file-based coordinate transfer used by `sub_main01/02` and `send_coords.py`
was replaced by the direct socket protocol in §6.6. The text-file approach had
no acknowledgement, no failure signalling, and no way to time a cycle — all
three are required for Experiment B.

---

## Citation

```bibtex
@inproceedings{kulhandjian2026strawberry,
  title     = {AI-Powered Strawberry Harvester Using a Robotic Arm},
  author    = {Kulhandjian, Hovannes and Khudayberdiev, Nursultan},
  booktitle = {Proc. IEEE},
  year      = {2026}
}
```

Related work from this group: AI-Powered Fruit Harvesting System Using a Robotic
Arm for Precision Agriculture (IEEE ICNC 2025); SARDOG (IEEE ICNC 2024);
AI-based Fruit Harvesting using a Robotic Arm (Int. Conf. Precision Agriculture
2024).