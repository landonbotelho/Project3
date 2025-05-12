import os
import math
import random
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import face_recognition

class FaceExplorer(Node):
    def __init__(self):
        super().__init__('project3v2_node')

        # Load and encode target face
        self.target_face = face_recognition.load_image_file(
            "/home/landon/turtlebot4_ws/src/project3_package/landon.jpg")
        self.target_encoding = face_recognition.face_encodings(self.target_face)[0]
        self.known_face_codes = [self.target_encoding]
        self.known_names = ["Landon"]

        # ROS2 setup
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/comp70/oakd/rgb/preview/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(Twist, '/comp70/cmd_vel', 10)

        # Movement state
        self.face_found = False
        self.face_center_x = None
        self.frame_width = 640
        self.backing_up = False
        self.backup_counter = 0
        self.stopped = False
        self.following_face = False

        # Movement speeds
        self.linear_speed = 0.25
        self.angular_speed = 1.2

        # Main movement loop
        self.timer = self.create_timer(0.3, self.move_robot)

    def image_callback(self, msg):
        if self.stopped:
            return  # Skip if permanently stopped

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame_width = frame.shape[1]
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        found_face_names = []
        self.face_center_x = None

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_face_codes, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_codes, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                found_face_names.append(self.known_names[best_match_index])
                self.face_center_x = (left + right) // 2

        # Draw boxes for visualization
        for (top, right, bottom, left), name in zip(face_locations, found_face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os._exit(0)

        # Start following behavior only once
        if "Landon" in found_face_names and not self.face_found and not self.stopped:
            self.get_logger().info("Target face detected. Following for 20 seconds...")
            self.face_found = True
            self.following_face = True
            self.create_timer(20.0, self.stop_completely)  # Stop after 20s

    def move_robot(self):
        twist = Twist()

        if self.stopped:
            self.cmd_pub.publish(twist)  # Stay stopped
            return

        # Handle backup mode
        if self.backing_up:
            if self.backup_counter < 10:
                twist.linear.x = -0.15
                twist.angular.z = random.choice([-1.0, 1.0])
                self.backup_counter += 1
                self.cmd_pub.publish(twist)
                return
            else:
                self.backing_up = False
                self.backup_counter = 0

        # Face-following behavior
        if self.following_face and self.face_center_x is not None:
            error = (self.frame_width // 2) - self.face_center_x
            twist.linear.x = 0.2
            twist.angular.z = float(error) / 100.0  # Proportional steering
            self.get_logger().info("Following face...")
        elif not self.face_found:
            # Random exploration
            twist.linear.x = self.linear_speed
            twist.angular.z = random.uniform(-self.angular_speed, self.angular_speed)

        self.cmd_pub.publish(twist)

    def stop_completely(self):
        if not self.stopped:
            self.get_logger().info("Stopping permanently after face follow.")
            self.stopped = True
            self.following_face = False
            self.stop_robot()

    def stop_and_backup(self):
        if not self.stopped:
            self.backing_up = True
            self.backup_counter = 0

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = FaceExplorer()

    # Only trigger obstacle aversion if not following face
    def obstacle_timer():
        if not node.stopped and not node.following_face:
            node.get_logger().info("Assuming bump: initiating backup.")
            node.stop_and_backup()

    node.create_timer(5.0, obstacle_timer)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
