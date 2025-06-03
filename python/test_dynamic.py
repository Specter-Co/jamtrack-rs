
import numpy as np
from byte_tracker_py import ByteTrackerPy
from byte_tracker import ByteTracker, Object, PyRect

class SimulatedDetection:
    image_w: int
    image_h: int
    w: float # bbox width in px
    h: float # bbox height in px
    x: float # top left x
    y: float # top left y

    def __init__(
        self,
        image_w: int = 1920,
        image_h: int = 1080,
        box_w_min: int = 50,
        box_w_max: int = 120,
        box_h_min: int = 200,
        box_h_max: int = 300,
        pad_perc: float = 0,
    ):
        self.image_w = image_w
        self.image_h = image_h
        
        self.x = np.random.uniform(0, image_w * 1.1)
        self.y = np.random.uniform(0, image_h * 1.1)

        self.vx = 0.5
        self.vy = 0.0
        
        self.w = np.random.uniform(box_w_min, box_w_max)
        self.h = np.random.uniform(box_h_min, box_h_max)
        self.score = 1 #np.random.uniform(0,1)

        self.pad_perc = pad_perc

    def update(self):
        # # First-order autoregressive models
        # nu = np.random.uniform(-1,1)
        # self.x = self.x + 5 * self.vx + 0.5 * nu
        # self.y = self.y + 5 * self.vy + 0.5 * nu

        # Deterministic model
        self.x += self.vx
        self.y += self.vy 

        # High score exercises expensive association
        self.score = 1.

        # Bounce off padded image frame
        if self.x < 0 - self.image_w * self.pad_perc:
            self.vx = np.abs(self.vx)
        if self.y < 0 - self.image_h * self.pad_perc:
            self.vy = np.abs(self.vy)
        if self.x + self.w > self.image_h * (1 + self.pad_perc):
            self.vx = -np.abs(self.vx)
        if self.y + self.h > self.image_h * (1 + self.pad_perc):
            self.vy = -np.abs(self.vy)

    def is_not_detected(self):
        is_out_of_frame = (
            (self.x + self.w < 0) |
            (self.x > self.image_w) | 
            (self.y > self.image_h) |
            (self.y + self.h < 0)
        )
        is_positive_score = self.score <= 0
        return is_out_of_frame or is_positive_score
        
    def get_tlbr(self):
        x_min = np.clip(self.x, 0, self.image_w)
        y_min = np.clip(self.y, 0, self.image_h)
        x_max = np.clip(self.x + self.w, 0, self.image_w)
        y_max = np.clip(self.y + self.h, 0, self.image_h)
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def get_score(self):
        return np.clip(self.score, 0, 1)

    def get_mot_format(self):
        return [0, 0, self.x, self.y, self.w, self.h, self.score, 0, 0, 0]

sim_detection = SimulatedDetection()
byte_tracker = ByteTrackerPy(
    frame_rate=10,
    track_buffer=5,
    track_thresh=0.5,
    high_thresh=0.7,
    match_thresh=0.7
)

for i in range(10):
    dets = [sim_detection.get_mot_format()]
    outputs = byte_tracker.update(dets)
    sim_detection.update()
    print("Vel XY: ", outputs[0].track_vel_xy, "Det ID:", outputs[0].detection_id)
    print()
