# Final Topic Subscription Corrections

## Issue Identified
The robot odometry topic `/turtlebot{i}/odom_map` publishes `nav_msgs/Odometry` messages, not `geometry_msgs/PoseStamped` messages.

## Changes Made

### 1. Updated Import Statement
```python
# Added nav_msgs.msg import
from nav_msgs.msg import Odometry
```

### 2. Corrected Robot Pose Subscription
```python
# Changed from PoseStamped to Odometry
self.robot_pose_sub = self.node.create_subscription(
    Odometry,  # Changed from PoseStamped
    f'/turtlebot{self.namespace_number}/odom_map',
    self.robot_pose_callback,
    10
)
```

### 3. Updated Robot Pose Callback
```python
def robot_pose_callback(self, msg):
    """Callback for robot pose updates - handles Odometry message"""
    self.robot_pose = msg.pose.pose  # Extract Pose from Odometry
```

### 4. Enhanced Distance Calculation
```python
def calculate_distance(self, pose1, pose2):
    """Calculate Euclidean distance between two poses"""
    if pose1 is None or pose2 is None:
        return float('inf')
    
    # Handle different pose message types
    # pose1 could be from Odometry (robot_pose) - extract position directly
    # pose2 could be from PoseStamped (relay/parcel) - extract from .pose.position
    if hasattr(pose1, 'pose'):
        # This is a PoseStamped message
        pos1 = pose1.pose.position
    else:
        # This is already a Pose message (from Odometry.pose.pose)
        pos1 = pose1.position
        
    if hasattr(pose2, 'pose'):
        # This is a PoseStamped message
        pos2 = pose2.pose.position
    else:
        # This is already a Pose message
        pos2 = pose2.position
    
    dx = pos1.x - pos2.x
    dy = pos1.y - pos2.y
    return math.sqrt(dx*dx + dy*dy)
```

## Final Topic Mapping

| Robot Namespace | Robot Pose Topic | Relay Pose Topic | Parcel Pose Topic | Current Index Topic |
|----------------|-------------------|-------------------|-------------------|-------------------|
| `tb0` | `/turtlebot0/odom_map` (Odometry) | `/Relaypoint0/pose` (PoseStamped) | `/parcel{i}/pose` (PoseStamped) | `/tb0/current_parcel_index` (Int32) |
| `tb1` | `/turtlebot1/odom_map` (Odometry) | `/Relaypoint1/pose` (PoseStamped) | `/parcel{i}/pose` (PoseStamped) | `/tb1/current_parcel_index` (Int32) |
| `tb2` | `/turtlebot2/odom_map` (Odometry) | `/Relaypoint2/pose` (PoseStamped) | `/parcel{i}/pose` (PoseStamped) | `/tb2/current_parcel_index` (Int32) |

## Verification Results

✅ **Robot Pose**: Successfully receiving from `/turtlebot0/odom_map`  
✅ **Relay Pose**: Successfully receiving from `/Relaypoint0/pose`  
✅ **Parcel Subscription**: Correctly subscribing to `/parcel0/pose`  
✅ **Message Handling**: Properly handling different message types in distance calculation  
✅ **Namespace Mapping**: Correct mapping `tb{i}` → `Relaypoint{i}` (not `i+1`)  

The WaitAction class is now fully compatible with the actual ROS2 topic structure and message types used in the system.
