import cv2
import numpy as np
from extract_pose import extract_landmarks_from_video

def make_square_with_padding(frame, size=640):
    """
    Returns:
      square_frame: (size,size,3)
      scale: float
      left, top: offsets in pixels inside square_frame
      new_w, new_h: resized frame size inside square
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    left = (size - new_w) // 2
    top = (size - new_h) // 2

    square = cv2.copyMakeBorder(
        resized,
        top, size - new_h - top,
        left, size - new_w - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return square, scale, left, top, new_w, new_h


def remap_landmarks_to_square(lm, orig_w, orig_h, scale, left, top, size=640):
    """
    lm: (33,4) with x,y normalized to ORIGINAL frame
    returns lm2: (33,4) with x,y normalized to SQUARE frame
    """
    lm2 = lm.copy()
    x = lm[:, 0] * orig_w
    y = lm[:, 1] * orig_h

    x2 = x * scale + left
    y2 = y * scale + top

    lm2[:, 0] = x2 / size
    lm2[:, 1] = y2 / size
    return lm2


CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]

def draw_pose(frame, lm, v_min=0.5):
    h, w = frame.shape[:2]

    def ok(i):
        x, y, z, v = lm[i]
        return (not np.isnan(x)) and (not np.isnan(y)) and (v >= v_min)

    # draw lines
    for a, b in CONNECTIONS:
        if ok(a) and ok(b):
            ax, ay = int(lm[a,0]*w), int(lm[a,1]*h)
            bx, by = int(lm[b,0]*w), int(lm[b,1]*h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)

    # draw points
    for i in range(33):
        if ok(i):
            cx, cy = int(lm[i,0]*w), int(lm[i,1]*h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)  # yellow points

# рисуем точки
def draw_landmarks(frame, lm):
    h, w = frame.shape[:2]
    for i in range(33):
        x, y, z, v = lm[i]
        if np.isnan(x) or np.isnan(y):
            continue
        cx, cy = int(x * w), int(y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

def main():
    video_path = r"0102(6).mp4"
    stride = 2
    size = 640

    seq = extract_landmarks_from_video(video_path, max_frames=180, stride=stride)
    cap = cv2.VideoCapture(video_path)

    idx = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # делаем квадратный кадр (без искажений)
        square_frame, scale, left, top, new_w, new_h = make_square_with_padding(frame, size=size)

        if frame_idx % stride == 0 and idx < len(seq):
            # переносим landmarks под квадратный кадр
            lm_sq = remap_landmarks_to_square(seq[idx], orig_w, orig_h, scale, left, top, size=size)

            draw_pose(square_frame, lm_sq, v_min=0.5)
            idx += 1

        frame_idx += 1
        cv2.imshow("pose preview", square_frame)  # показываем квадрат
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()