import numpy as np
import cv2
from abc import abstractmethod
from drawing_tools import draw_velocity_vector, generate_grid_segments
"""
Helper functions and classes for alert geometry.

Ie determining if a track should be alerted based on a given alert geometric structure

Current supported geometry:
- Line segment
- Quadrilateral
"""

def compute_transform(start, end):
    # Homography can be estimated from 4 point correspondences
    transform_constraints = []
    for (x, y), (u, v) in zip(start, end):
        transform_constraints.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        transform_constraints.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    transform_constraints += [[0,0,0,0,0,0,0,0,1]]

    transform_constraints = np.array(transform_constraints)

    # solve transform_constraints * homography_coefficients = 0
    return np.linalg.solve(transform_constraints, np.array([0,0,0,0,0,0,0,0,1])).reshape(3,3)

def apply_transform(transform, point):
    homogenous_point = np.array([point[0], point[1], 1]).T
    homogenous_solution = transform @ homogenous_point
    return [homogenous_solution[0] / homogenous_solution[2], homogenous_solution[1] / homogenous_solution[2]]

def horizon_check(transform, point, end_points):
    """
    Returns true if the point is above the horizon.
    """
    homogenous_point = np.array([point[0], point[1], 1]).T
    homogenous_solution = transform @ homogenous_point
    point_sign = np.sign(homogenous_solution[2])
    
    x_mean = 0
    y_mean = 0
    for p in end_points:
        x_mean += 1/4 * (p[0])
        y_mean += 1/4 * (p[1])

    homogenous_centroid = np.array([x_mean, y_mean, 1]).T
    homogenous_solution = transform @ homogenous_centroid
    centroid_sign = np.sign(homogenous_solution[2])

    if centroid_sign * point_sign < 0:
        return True
    else:
        return False
    
def rotate_2d(x, angle_rad):
    r = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)],
         [np.sin(angle_rad), np.cos(angle_rad)]]
    )
    return r @ x

def segments_intersect(p1, p2, q1, q2):
    """
    Check if two line segments (p1 to p2) and (q1 to q2) intersect.

    Args:
        p1, p2: Tuples (x, y) defining the first segment.
        q1, q2: Tuples (x, y) defining the second segment.

    Returns:
        bool: True if segments intersect, False otherwise.
    """
    def ccw(a, b, c):
        # Check if points a, b, c are listed in a counterclockwise order
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """
    Compute the shortest distance from a point to a line segment.

    Args:
        px, py (float): Coordinates of the point.
        x1, y1 (float): Start of the segment.
        x2, y2 (float): End of the segment.

    Returns:
        float: Distance from point to the line segment.
    """
    # Line segment vector
    dx = x2 - x1
    dy = y2 - y1

    # Handle degenerate segment (start == end)
    if dx == 0 and dy == 0:
        return np.hypot(px - x1, py - y1)

    # Project point onto the segment line (parametric t)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp to segment

    # Nearest point on segment
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    # Distance to nearest point
    dist = np.hypot(px - nearest_x, py - nearest_y)
    return dist

def is_point_on_correct_side(d, point, line_start):
    # Vector from line_start to point (AP)
    AP = [point[0] - line_start[0], point[1] - line_start[1]]
    
    # If the cross products have the same sign (both positive or both negative),
    # the point is on the correct side of the line.
    return AP[0] * d[0] + AP[1] * d[1] >= 0

def is_point_inside_quad(quad_points, point):
    assert len(quad_points) == 4
    for i in range(4):
        pt1 = quad_points[i]
        pt2 = quad_points[(i + 1) % 4]
        # Normal to edge
        d_edge_norm = pt2[1] - pt1[1], -(pt2[0] - pt1[0])
        if not is_point_on_correct_side(d_edge_norm, point, pt1):
            return False
    return True
        
class BaseAlertGeometry:
    def __init__(self, geom_type, threshold_points, radius):
        self.geom_type = geom_type
        self.threshold_points = threshold_points
        pt1, pt2 = threshold_points[0], threshold_points[1]
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        self.edge = pt1, pt2
        self.edge_norm = -dy, dx
        self.radius = radius

    @abstractmethod
    def draw_threshold(self, img, color=(255,0,255), thickness=2):
        # Draw flow edge
        pt1, pt2 = self.edge
        dx, dy = self.edge_norm
        cv2.line(img, pt1, pt2, color=color, thickness=thickness)
        # Plot normal vector
        draw_velocity_vector(img, (pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2, dx, dy, scale=0.1)
    
    @abstractmethod
    def get_track_status(self, track_box, track_v, track_status, frame):
        pass

class LineAlertGeometry(BaseAlertGeometry):
    def __init__(self, threshold_points, radius):
        assert len(threshold_points) == 2, f"Line alert geometry expects 2 threshold points but got {len(threshold_points)}"
        super().__init__('line', threshold_points, radius)

    def get_track_status(self, track_pos, track_v, track_status, frame, track_prev_pos=None):
        """
        Given a track determine the new track status. 
        Track position is defined by the bottom middle of the box.

        Alg:
        - Determine if track crosses edge by checking if curr_pos - prev_pos intersects the edge
        - Determine if track is going against flow direction by getting inner prod between track vel and flow dir
        - If track is going against direction
            - track crosses edge -> set alert
            - otherwise if track is within radius -> set warning

        Returns track status
        0 = Normal
        1 = Alert
        -1 = Warning
        """
        # Persist alert
        if track_status == 1:
            return 1
        
        vx, vy = track_v # From kalman, bbox center vel
        # Geometry edge
        pt1, pt2 = self.edge
        # Flow direction
        dx, dy = self.edge_norm
        # Is track going against flow direction
        is_against_flow = vx * dx + vy * dy < 0

        if not is_against_flow:
            return 0
        if track_prev_pos is not None and segments_intersect(track_prev_pos, track_pos, pt1, pt2):
            # Check if track crossed edge between prev and curr location
            return 1
        # If track is normal status and hasn't crossed edge, 
        # check if we should set warning if the class is too close to edge and going wrong way
        if (
            track_status == 0 and
            point_to_segment_distance(track_pos[0], track_pos[1], pt1[0], pt1[1], pt2[0], pt2[1]) < self.radius and
            is_point_on_correct_side((dy, dx), track_pos, pt1)
        ):
            return -1
        return 0

class QuadAlertGeometry(BaseAlertGeometry):
    """
    Threshold must be passed in in BL,BR,TR,TL since the BEV transform assumes this ordering!
    # TODO ryan, should I do a smart sort or something to remove this requirement?
    Used 'edge' to determine what edge should determine flow. 
    The correct direction of flow will always point outside the quadrilateral

    See these slides for more details on algorithm
    https://docs.google.com/presentation/d/1D5kpt3aXJ_eTBgg-3SnhLu-2xNAE-Vmuyd_w11tlSMc/edit?slide=id.g35abc9ec82a_1_0#slide=id.g35abc9ec82a_1_0
    """
    def __init__(self, threshold_points, radius, edge_idx=[0,1], bev_h=1, bev_w=1):
        assert len(threshold_points) == 4, f"Quad alert geometry expects 4 threshold points but got {len(threshold_points)}"
        # User selected alert region
        self.alert_region = np.stack(threshold_points)
        
        # Edge points should be ordered ('counter-clockwise') so that edge points outwards from the quadrilateral
        def make_clockwise(edge_idx):
            if edge_idx[0] == 0 and edge_idx[1] == 3:
                return [3,0]
            return sorted(edge_idx)
        self.edge_idx = make_clockwise(edge_idx)

        # BEV parallelogram
        self.bev_h, self.bev_w = bev_h, bev_w
        # BL, BR, TR, TL
        self.bev_rect = np.array([[0, bev_h], [bev_w, bev_h], [bev_w, 0], [0, 0]])
        # Homography to transform from Projected space to BEV space
        self.H = compute_transform(self.bev_rect, self.alert_region)
        # Homography to transform from BEV space to Projected space
        self.H_inv = compute_transform(self.alert_region, self.bev_rect)

        super().__init__('quad', (threshold_points[self.edge_idx[0]], threshold_points[self.edge_idx[1]]), radius)
    
    def draw_threshold(self, img, color=(255,0,255), thickness=2, draw_grid_lines=True):
        # Draws flow edge
        super().draw_threshold(img, color=color, thickness=thickness)
        drawn_points = set(self.edge_idx)
        # Draw rest of quadrilateral
        for i in range(4):
            # Dont re-draw the flow edge we already drew
            j = (i + 1) % 4
            
            if i in drawn_points and j in drawn_points:
                continue
            
            pt1, pt2 = self.alert_region[i], self.alert_region[j] 
            cv2.line(img, pt1, pt2, color=color, thickness=thickness)

        if draw_grid_lines:
            grid_lines = generate_grid_segments(n=10)
            for p1, p2 in grid_lines:
                x0,y0 = apply_transform(self.H, p1)
                x1,y1 = apply_transform(self.H, p2)
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color=(0,0,0), thickness=1)

    def get_track_status(self, track_pos, track_v, track_status, img, debug_plotting=True):
        """
        - track_pos: (x, y) in projected space
        - track_v: (vx, vy) in projected space
        - track_status: 0 = normal, 1 = alert, -1 = warning of the tracks previous status
        - img: image to draw on
        - debug_plotting: if True, will draw debug lines

        Returns track status
        """

        in_quad = is_point_inside_quad(self.alert_region, track_pos)
            
        end_points = self.alert_region
        
        # Check if track pos is above horizon
        if horizon_check(self.H_inv, track_pos, end_points):
            # print("Point is above the horizon. Cannot continue. Returning 2.")
            return 2, 0, in_quad

        # p' (query point in BEV space)
        i_query = apply_transform(self.H_inv, track_pos)

        # Query lines in BEV space
        # Structure:
        # [(line 1), (line 2)]
        # [((x0, y0), (x1, y1)), ((x2, y2), (x3, y3))]
        i_query_lines = [((0., i_query[1]), (1., i_query[1])), ((i_query[0], 0.), (i_query[0], 1.))]
        
        # Edge in Projected space
        edge_points = [end_points[self.edge_idx[0]], end_points[self.edge_idx[1]]]

        # Edge in BEV space
        i_edge_points = [apply_transform(self.H_inv, p) for p in edge_points]

        # Determine whether u' or v' is parallel to the edge in BEV space, call this t'
        (u0_prime, u1_prime) , (v0_prime, v1_prime) = i_query_lines
        i_edge = np.array(i_edge_points[1]) - np.array(i_edge_points[0])

        u_prime = np.array(u1_prime) - np.array(u0_prime)
        v_prime = np.array(v1_prime) - np.array(v0_prime)
        abs_dot_u = np.abs(np.dot(u_prime, i_edge))
        abs_dot_v = np.abs(np.dot(v_prime, i_edge))
        t_prime_points = (u0_prime, u1_prime) if abs_dot_u > abs_dot_v else (v0_prime, v1_prime)
        t_prime = u_prime if abs_dot_u > abs_dot_v else v_prime

        # Determine whether t_prime must be rotated +90 or -90 to 
        # Find edge f (opposite of the user selected edge)
        f_edge_idx = [
            (self.edge_idx[0] + 2) % 4,
            (self.edge_idx[1] + 2) % 4,
        ]
        f_edge_points = [end_points[f_edge_idx[0]], end_points[f_edge_idx[1]]]
        # Note: q_prime points in direction of permitted flow
        q_prime = np.array(edge_points[1]) - np.array(f_edge_points[0])
        x_sign = np.sign(np.cross(t_prime, q_prime))

        # Transform t from BEV space to Projected space
        t_points = [apply_transform(self.H, p) for p in t_prime_points]

        # Compute perpendicular vectors w1 and w2
        t0, t1 = t_points
        t = np.array(t1) - np.array(t0)
        # Vector that determines correct flow dir
        w = rotate_2d(t, x_sign * np.pi / 2)
        w_dx, w_dy = (np.array(w) / np.linalg.norm(np.array(w))).tolist()
        
        vx, vy = track_v # From kalman, bbox center vel
        # Project vel & flow_dir into BEV spae
        x2, y2 = track_pos[0] + vx, track_pos[1] + vy
        x2_prime, y2_prime = apply_transform(self.H_inv, (x2, y2))
        x1_prime, y1_prime = i_query
        t0_x_prime, t0_y_prime = apply_transform(self.H_inv, (t0[0], t0[1]))
        t1_x_prime, t1_y_prime = apply_transform(self.H_inv, (t1[0], t1[1]))
        vx_prime, vy_prime = (x2_prime - x1_prime), (y2_prime - y1_prime)
        wx_prime, wy_prime = (t1_x_prime - t0_x_prime), (t1_y_prime - t0_y_prime)
        # Normalize w_prime
        wx_prime, wy_prime = (np.array((wx_prime, wy_prime)) / np.linalg.norm(np.array((wx_prime, wy_prime)))).tolist()
        # Inner prod in BEV space
        # inner_prod = vx_prime * wx_prime + vy_prime * wy_prime

        # Is track going against flow direction
        inner_prod = vx * w_dx + vy * w_dy
        is_against_flow = inner_prod < 0

        if debug_plotting:
            # Plot track allowed flow dir
            color = (255,0,0) if is_against_flow else (0,255,0)
            draw_velocity_vector(img, track_pos[0], track_pos[1], w_dx, w_dy, color=color, scale=50)
            # Plot u,v lines
            if in_quad:
                for p1, p2 in i_query_lines:
                    x0,y0 = apply_transform(self.H, p1)
                    x1,y1 = apply_transform(self.H, p2)
                    cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color=(0,0,0), thickness=1)

        # Persist alert
        if track_status == 1 and in_quad:
            return 1, inner_prod, in_quad, (w_dx, w_dy)
        if not is_against_flow:
            return 0, inner_prod, in_quad, (w_dx, w_dy)
        # Alert
        if in_quad and is_against_flow:
            return 1, inner_prod, in_quad, (w_dx, w_dy)
        # Warning
        if track_status == 0 and not in_quad:
            for i in range(4):
                pt1, pt2 = self.alert_region[i], self.alert_region[(i + 1) % 4]
                dist = point_to_segment_distance(track_pos[0], track_pos[1], pt1[0], pt1[1], pt2[0], pt2[1])
                if dist < self.radius:
                    return -1, inner_prod, in_quad, (w_dx, w_dy)
        return 0, inner_prod, in_quad, (w_dx, w_dy)

        


