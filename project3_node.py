import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import qos_profile_sensor_data
import face_recognition
import numpy
#merging camera subscriber and face recognition example
#https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py#L3
class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.target_face = face_recognition.load_image_file("/home/landon/turtlebot4_ws/src/landon.jpg")
        self.target_encoding= face_recognition.face_encodings(self.target_face)[0]
        self.known_face_codes = [
            self.target_encoding
        ]
        self.known_names = [ #what a hack make sure face code list and names list is in same order
            "Landon"
        ]
        self.subscription = self.create_subscription(
            Image,
            '/comp70/oakd/rgb/preview/image_raw',
            self.listener_callback,
            qos_profile_sensor_data)
        self.bridge = CvBridge()
        

    def listener_callback(self, msg):
        robot_view = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_robot_view = numpy.ascontiguousarray(robot_view[:, :, ::-1])
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_robot_view)
        # debugging print(f"{len(face_locations)} found!\nface locations: {face_locations}")
        face_encodings = face_recognition.face_encodings(rgb_robot_view, face_locations)
        found_face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces( self.known_face_codes, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_codes, face_encoding)
            best_match_index = numpy.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_names[best_match_index]

            found_face_names.append(name)
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, found_face_names):
            # Draw a box around the face
            cv2.rectangle(robot_view, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(robot_view, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(robot_view, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Camera Feed', robot_view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os.exit(0)

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
