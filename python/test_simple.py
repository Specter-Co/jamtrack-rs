from byte_tracker import ByteTracker, Object, PyRect
from typing import List

# Create a ByteTracker
tracker = ByteTracker(
    frame_rate=30,
    track_buffer=30,
    track_thresh=0.5,
    high_thresh=0.6,
    match_thresh=0.8
)

# Create PyRect instances
rect1 = PyRect(x=100.0, y=100.0, width=50.0, height=50.0)
rect2 = PyRect(x=200.0, y=200.0, width=60.0, height=60.0)

# Create Object instances
obj1 = Object(
    detection_id=1,
    rect=rect1,
    prob=0.9,
    track_id=None,
    track_vel_xy=None
)
obj2 = Object(
    detection_id=1,
    rect=rect2,
    prob=0.7,
    track_id=None,
    track_vel_xy=None
)

# Call update with a list of Objects
results: List[Object] = tracker.update([obj1, obj2])
results: List[Object] = tracker.update([obj1, obj2])

# Print results
for result in results:
    print(
        f"Prob: {result.prob}, "
        f"Rect: ({result.x}, {result.y}, {result.width}, {result.height}), "
        f"Rect: ({result.rect.x}, {result.rect.y}, {result.rect.width}, {result.rect.height}), "
        f"Det ID: {result.detection_id}, "
        f"Track ID: {result.track_id}, "
        f"Velocity: {result.track_vel_xy}"
    )

print(rect1.x)
