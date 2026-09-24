"""
Eye-to-hand calibration for a fixed RealSense D435 and a myCobot 320 Pi.

Configuration assumed here: the camera is bolted to the overhead boom and does
not move; a ChArUco board is mounted rigidly to the arm's tool flange. The
unknown being solved for is the transform from the camera frame to the robot
base frame.

Run modes:
    python handeye_calibrate.py           # AUTO: move arm, capture, solve, write cam2base.json
    python handeye_calibrate.py board     # generate a printable board
    python handeye_calibrate.py solve     # re-solve from an existing handeye_poses.json

AUTO mode needs only one thing from you: before running, leave the arm in a
pose where the camera can see the board (or set HOME_ANGLES below). The script
perturbs the joints around that pose, captures the board at each one, returns
home, solves with every OpenCV hand-eye method, drops outlier poses, and
writes cam2base.json.

Dependencies: opencv-contrib-python==4.10.0.84, pyrealsense2, numpy, ikpy, pymycobot.
"""

import sys
import json
import time
import numpy as np
import cv2

# OpenCV 5.x dropped calibrateHandEye and interpolateCornersCharuco from the
# Python bindings. Stay on the 4.x line for this script.
if not hasattr(cv2, "calibrateHandEye"):
    raise SystemExit(
        f"OpenCV {cv2.__version__} does not expose calibrateHandEye.\n"
        "Install a 4.x build:  pip install opencv-contrib-python==4.10.0.84"
    )


# =====================================================================
# Board definition -- EDIT THESE to match your printed board
# =====================================================================
SQUARES_X = 5              # number of chessboard squares across
SQUARES_Y = 7              # number of chessboard squares down
SQUARE_LENGTH_M = 0.030    # MEASURE THIS with a caliper after printing
MARKER_LENGTH_M = 0.022    # MEASURE THIS too; must be < SQUARE_LENGTH_M

ARUCO_DICT = cv2.aruco.DICT_5X5_100
POSE_FILE = "handeye_poses.json"
RESULT_FILE = "cam2base.json"

# =====================================================================
# Robot / automation settings
# =====================================================================
ROBOT_IP = "129.8.233.110"
ROBOT_PORT = 9000
URDF_PATH = "mycobot_320pi.urdf"

HOME_ANGLES = None          # None = use the arm's current pose as the centre pose
N_POSES = 20                # total poses including home
SPEED = 10                  # send_angles speed (1-100); keep low
SEED = 0                    # change to get a different pose set

# Max perturbation per joint (deg) around home. Shoulder/elbow small to avoid
# hitting the table or boom; wrist large because rotation variety is what
# makes the solve well-conditioned.
PERTURB_DEG = [10, 8, 8, 20, 20, 30]

# myCobot 320 joint limits with a 5 deg safety margin
JOINT_LIMITS = [(-165, 165), (-132, 132), (-146, 146),
                (-143, 143), (-164, 164), (-175, 175)]

ARRIVE_TOL_DEG = 2.0        # "arrived" when every joint is within this
MOVE_TIMEOUT_S = 15.0
SETTLE_S = 1.0              # wait after arrival for vibration to die out
FRAMES_PER_POSE = 8         # best of N frames is kept
MIN_CORNERS = 10            # of 24 inner corners on a 5x7 board
MAX_REPROJ_PX = 1.0         # reject PnP solutions worse than this
MIN_SAMPLES = 8


# ---------------------------------------------------------------------
# OpenCV 4.7 changed the aruco API. These wrappers work either way.
# ---------------------------------------------------------------------
def _get_dictionary():
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.Dictionary_get(ARUCO_DICT)


def _get_board(dictionary):
    if hasattr(cv2.aruco, "CharucoBoard") and not hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary
        )
    return cv2.aruco.CharucoBoard_create(
        SQUARES_X, SQUARES_Y, SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary
    )


def _detect_charuco(gray, board, dictionary):
    """Returns (charuco_corners, charuco_ids) or (None, None)."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is None or len(ids) < 4:
        return None, None

    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if retval is None or retval < 6:
        return None, None
    return ch_corners, ch_ids


def _board_object_points(board, ch_ids):
    """3D coordinates of the detected chessboard corners, board frame."""
    if hasattr(board, "getChessboardCorners"):
        all_pts = board.getChessboardCorners()
    else:
        all_pts = board.chessboardCorners
    return np.array([all_pts[i[0]] for i in ch_ids], dtype=np.float32)


# =====================================================================
# Mode: board -- write a printable PNG
# =====================================================================
def make_board(path="charuco_board.png", dpi=300):
    dictionary = _get_dictionary()
    board = _get_board(dictionary)

    w_px = int(SQUARES_X * SQUARE_LENGTH_M * 1000 / 25.4 * dpi)
    h_px = int(SQUARES_Y * SQUARE_LENGTH_M * 1000 / 25.4 * dpi)

    if hasattr(board, "generateImage"):
        img = board.generateImage((w_px, h_px))
    else:
        img = board.draw((w_px, h_px))

    cv2.imwrite(path, img)
    print(f"wrote {path}  ({w_px}x{h_px} px, print at {dpi} dpi, NO scaling)")
    print("After printing, measure one square and one marker with a caliper")
    print("and update SQUARE_LENGTH_M / MARKER_LENGTH_M in this file.")


# =====================================================================
# Robot helpers
# =====================================================================
def _get_angles(mc, retries=5):
    """get_angles() over the socket sometimes returns -1 or junk; retry."""
    for _ in range(retries):
        a = mc.get_angles()
        if isinstance(a, (list, tuple)) and len(a) == 6:
            return [float(x) for x in a]
        time.sleep(0.3)
    return None


def _move_and_wait(mc, target, speed=SPEED):
    """Command a pose, block until arrived (or stalled), return MEASURED angles."""
    mc.send_angles([float(x) for x in target], speed)
    t0 = time.time()
    last = None
    while time.time() - t0 < MOVE_TIMEOUT_S:
        time.sleep(0.25)
        a = _get_angles(mc, retries=2)
        if a is None:
            continue
        if max(abs(x - y) for x, y in zip(a, target)) < ARRIVE_TOL_DEG:
            break
        # stopped short of the target (limit, obstruction): accept where it is
        if last is not None and time.time() - t0 > 3 and \
                max(abs(x - y) for x, y in zip(a, last)) < 0.1:
            print("    arm stopped short of target; using actual pose")
            break
        last = a
    time.sleep(SETTLE_S)
    return _get_angles(mc)


def _make_poses(home, n, seed=SEED):
    """n poses around home; every joint moved by 40-100% of its range so the
    rotation axes differ from pose to pose."""
    rng = np.random.default_rng(seed)
    lo = np.array([l for l, _ in JOINT_LIMITS], dtype=float)
    hi = np.array([h for _, h in JOINT_LIMITS], dtype=float)
    span = np.array(PERTURB_DEG, dtype=float)
    poses = []
    for _ in range(n):
        off = rng.choice([-1.0, 1.0], 6) * rng.uniform(0.4, 1.0, 6) * span
        poses.append(np.clip(np.array(home) + off, lo, hi).round(2).tolist())
    return poses


# =====================================================================
# Camera helpers
# =====================================================================
def _start_camera():
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)

    for _ in range(30):                     # let auto-exposure settle
        pipeline.wait_for_frames()
    return pipeline, K, dist


def _capture_board(pipeline, board, dictionary, K, dist):
    """Best board pose over FRAMES_PER_POSE frames, or None."""
    best = None
    for _ in range(FRAMES_PER_POSE):
        cf = pipeline.wait_for_frames().get_color_frame()
        if not cf:
            continue
        gray = cv2.cvtColor(np.asanyarray(cf.get_data()), cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids = _detect_charuco(gray, board, dictionary)
        if ch_corners is None or len(ch_ids) < MIN_CORNERS:
            continue

        obj = _board_object_points(board, ch_ids)
        img = ch_corners.reshape(-1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        if err > MAX_REPROJ_PX:
            continue

        key = (len(ch_ids), -err)
        if best is None or key > best[0]:
            best = (key, rvec, tvec, err)

    if best is None:
        return None
    (n, _), rvec, tvec, err = best
    return {
        "rvec_target2cam": rvec.ravel().tolist(),
        "tvec_target2cam": tvec.ravel().tolist(),
        "n_corners": int(n),
        "reproj_px": round(err, 3),
    }


# =====================================================================
# Mode: auto collect -- arm drives itself through the poses
# =====================================================================
def collect_auto():
    from pymycobot import MyCobot320Socket

    dictionary = _get_dictionary()
    board = _get_board(dictionary)

    mc = MyCobot320Socket(ROBOT_IP, ROBOT_PORT)
    time.sleep(0.5)

    home = HOME_ANGLES or _get_angles(mc)
    if home is None:
        raise SystemExit("could not read joint angles. Restart Server_320.py on the Pi "
                         "(sudo fuser -k 9000/tcp) and try again.")
    print(f"home pose: {np.round(home, 1)}")

    pipeline, K, dist = _start_camera()
    samples = []
    try:
        home_actual = _move_and_wait(mc, home)
        if _capture_board(pipeline, board, dictionary, K, dist) is None:
            raise SystemExit("board not visible from the home pose. Move the arm so "
                             "the camera sees the board, then rerun.")

        targets = [home] + _make_poses(home, N_POSES - 1)
        for i, tgt in enumerate(targets, 1):
            print(f"[{i:2d}/{len(targets)}] -> {np.round(tgt, 1)}")
            actual = home_actual if i == 1 else _move_and_wait(mc, tgt)
            if actual is None:
                print("    no joint readback, skipping")
                continue
            det = _capture_board(pipeline, board, dictionary, K, dist)
            if det is None:
                print("    board not detected / poor fit, skipping")
                continue
            samples.append({"joint_angles_deg": actual, **det})
            print(f"    ok: {det['n_corners']} corners, reproj {det['reproj_px']} px")

        print("returning home")
        _move_and_wait(mc, home)
    finally:
        pipeline.stop()

    with open(POSE_FILE, "w") as f:
        json.dump({"K": K.tolist(), "dist": dist.tolist(), "samples": samples}, f, indent=2)
    print(f"\nwrote {len(samples)} samples to {POSE_FILE}")

    if len(samples) < MIN_SAMPLES:
        raise SystemExit(f"only {len(samples)} usable poses (need {MIN_SAMPLES}). "
                         "Reduce PERTURB_DEG or improve board visibility, then rerun.")


# =====================================================================
# Mode: solve
# =====================================================================
METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_chain(urdf_path=URDF_PATH):
    from ikpy.chain import Chain
    return Chain.from_urdf_file(
        urdf_path,
        active_links_mask=[False, True, True, True, True, True, True, False],
    )


def fk_pose(chain, angles_deg):
    """Forward kinematics -> 4x4 flange pose in the base frame."""
    q = np.zeros(len(chain.links))
    active = [i for i, a in enumerate(chain.active_links_mask) if a]
    for k, i in enumerate(active):
        q[i] = np.deg2rad(angles_deg[k])
    return chain.forward_kinematics(q)


def _solve_once(T_g2b, T_t2c, method):
    # eye-to-hand: feed the INVERSE of the flange pose
    R_b2g = [T[:3, :3].T for T in T_g2b]
    t_b2g = [(-T[:3, :3].T @ T[:3, 3]).reshape(3, 1) for T in T_g2b]
    R_t2c = [T[:3, :3] for T in T_t2c]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in T_t2c]

    R, t = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=method)
    T_c2b = np.eye(4)
    T_c2b[:3, :3] = R
    T_c2b[:3, 3] = t.ravel()

    # residual: board origin in the flange frame should be identical in every pose
    pts = np.array([(np.linalg.inv(Tg) @ T_c2b @ Tt)[:3, 3]
                    for Tg, Tt in zip(T_g2b, T_t2c)])
    centroid = pts.mean(axis=0)
    resid = np.linalg.norm(pts - centroid, axis=1)
    return T_c2b, resid, centroid


def _best_method(T_g2b, T_t2c):
    best = None
    for name, m in METHODS.items():
        try:
            T, r, c = _solve_once(T_g2b, T_t2c, m)
        except cv2.error:
            continue
        if not np.all(np.isfinite(T)):
            continue
        rms = float(np.sqrt((r ** 2).mean()))
        print(f"  {name:10s} RMS {rms * 1000:6.2f} mm")
        if best is None or rms < best[0]:
            best = (rms, name, T, r, c)
    if best is None:
        raise SystemExit("every hand-eye method failed; poses are degenerate")
    return best


def solve(urdf_path=URDF_PATH):
    with open(POSE_FILE) as f:
        samples = json.load(f)["samples"]
    if len(samples) < MIN_SAMPLES:
        raise SystemExit(f"only {len(samples)} samples; need at least {MIN_SAMPLES}")

    chain = load_chain(urdf_path)
    T_g2b = [fk_pose(chain, s["joint_angles_deg"]) for s in samples]
    T_t2c = []
    for s in samples:
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(np.array(s["rvec_target2cam"], dtype=np.float64))[0]
        T[:3, 3] = s["tvec_target2cam"]
        T_t2c.append(T)

    print(f"\nsolving with {len(samples)} poses:")
    rms, name, T_c2b, resid, centroid = _best_method(T_g2b, T_t2c)

    # one round of outlier rejection
    dropped = []
    thresh = max(3 * np.median(resid), 0.003)
    bad = resid > thresh
    if bad.any() and (~bad).sum() >= MIN_SAMPLES:
        dropped = np.flatnonzero(bad).tolist()
        keep = np.flatnonzero(~bad)
        print(f"\ndropping {len(dropped)} outlier pose(s) {dropped}, re-solving:")
        rms, name, T_c2b, resid, centroid = _best_method(
            [T_g2b[i] for i in keep], [T_t2c[i] for i in keep])

    print(f"\nbest method: {name}")
    print("camera -> base transform (4x4, metres):")
    print(np.array2string(T_c2b, precision=5, suppress_small=True))
    print(f"camera position in base frame: {np.round(T_c2b[:3, 3] * 1000, 1)} mm")
    print(f"\nresidual over {len(resid)} poses:")
    print(f"  RMS  {rms * 1000:6.2f} mm")
    print(f"  mean {resid.mean() * 1000:6.2f} mm")
    print(f"  max  {resid.max() * 1000:6.2f} mm")
    print(f"  board origin in flange frame: {np.round(centroid * 1000, 1)} mm")

    if rms * 1000 > 5:
        print("\n  RMS above 5 mm -- likely causes: board small in frame (640x480),")
        print("  too little orientation variation, board loose on the flange,")
        print("  or a wrong SQUARE_LENGTH_M.")

    with open(RESULT_FILE, "w") as f:
        json.dump({
            "T_cam2base": T_c2b.tolist(),
            "residual_rms_mm": rms * 1000,
            "n_poses": int(len(resid)),
            "method": name,
            "dropped_pose_indices": dropped,
        }, f, indent=2)
    print(f"\nwrote {RESULT_FILE}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    if mode == "auto":
        collect_auto()
        solve()
    elif mode == "board":
        make_board()
    elif mode == "solve":
        solve(*sys.argv[2:])
    else:
        print(__doc__)