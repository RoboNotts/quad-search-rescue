#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pyzbar.pyzbar import decode

bridge = CvBridge()

def image_callback(msg):
    try:
        # Convert ROS Image to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Process image (optional, e.g., convert to grayscale)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect QR codes
        qr_codes = decode(gray_image)    	
        for qr in qr_codes:
            (x, y, w, h) = qr.rect
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            qr_data = qr.data.decode("utf-8")
            print(f"QR Code Detected: {qr_data}")
            
            cv2.putText(cv_image, qr_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image
        cv2.imshow("QR Code Scanner", cv_image)
        #cv2.imshow("Grayscale", gray_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User exit")
    except Exception as e:
        rospy.logerr("Error processing image: %s", str(e))

if __name__ == "__main__":
    rospy.init_node("realsense_rgb_listener", anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rospy.loginfo("Subscribed to /camera/color/image_raw")
    rospy.spin()
    cv2.destroyAllWindows()
