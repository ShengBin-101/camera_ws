import rclpy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Float32

# This node subscribes to the left and right camera topics and publishes the estimated distance to a yellow object

# baseline of camera is 13.8 cm
baseline = 0.138 

# focal length is 2.96mm (according to specsheet)
focal_length_mm = 2.96

# pixel size in mm
pixel_size_mm = 1.12 / 1000  # convert from μm to mm

# sensor resolution in pixels
sensor_resolution_pixels = 3280  # width of the sensor active area

# sensor width in mmfocal_length_mm
sensor_width_mm = sensor_resolution_pixels * pixel_size_mm

# calculate pixels per mm
pixels_per_mm = sensor_resolution_pixels / sensor_width_mm

# focal_length_pixels = focal_length_mm * pixels_per_mm
focal_length_pixels_1 = (3280*0.5) / (np.tan(50.7 * 0.5 * np.pi / 180))

# Converting focal length in 2.96mm to focal length in pixels 
focal_length_pixels_2 = focal_length_mm * pixels_per_mm

class StereoVisionNode:
    def __init__(self):
        self.node = rclpy.create_node('stereo_vision_node')
        self.left_sub = self.node.create_subscription(
            CompressedImage, "/left/calibrated/compressed", self.left_image_callback, 10)
        self.right_sub = self.node.create_subscription(
            CompressedImage, "/right/calibrated/compressed", self.right_image_callback, 10)
        self.disparity_pub = self.node.create_publisher(
            CompressedImage, "/left/disparity/compressed", 10)
        self.depthimg_pub = self.node.create_publisher(
            CompressedImage, "/left/depthimg/compressed", 10)
        
        self.bridge = CvBridge()

    def left_image_callback(self, left_msg):
        self.left_image = self.bridge.compressed_imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        self.process_images()

    def right_image_callback(self, right_msg):
        self.right_image = self.bridge.compressed_imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        self.process_images()

    def process_images(self):
        if hasattr(self, 'left_image') and hasattr(self, 'right_image'):
            
            # Optional: Segment regions of interest

            # Compute disparity map
            disparity_map = self.compute_disparity(self.left_image, self.right_image)

            # Convert disparity map to depth map
            depth_map = self.convert_disparity_to_depth(disparity_map)

            # Estimate distance to yellow object
            distance = self.estimate_distance(depth_map)
            print(distance)

            # Convert disparity map to 8-bit image
            disparity_map = cv2.normalize(
                disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

            # Publish the disparity map
            disparity_msg = self.bridge.cv2_to_compressed_imgmsg(disparity_map)
            self.disparity_pub.publish(disparity_msg)

            # Convert depth map to 8-bit image

            depth_map = depth_map.astype(np.uint8)
            depth_map = cv2.normalize(
                depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

            # Publish the depth map
            depth_msg = self.bridge.cv2_to_compressed_imgmsg(depth_map)
            self.depthimg_pub.publish(depth_msg)

    def convert_disparity_to_depth(self, disparity_map):
        # Convert disparity map to depth map, ignoring pixels with value 0
        depth_map = np.zeros(disparity_map.shape)
        depth_map[disparity_map > 0] = (focal_length_pixels_1 * baseline) / disparity_map[disparity_map > 0]

        return depth_map

    def compute_disparity(self, left_image, right_image):
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM_create()

        stereo.setMinDisparity(5)
        stereo.setNumDisparities(32)
        stereo.setBlockSize(41)
        stereo.setSpeckleRange(9)
        stereo.setSpeckleWindowSize(12)
        stereo.setDisp12MaxDiff(3)
        stereo.setUniquenessRatio(5)
        stereo.setPreFilterCap(16)
        stereo.setPreFilterSize(9)
        stereo.setPreFilterType(1)
        stereo.setTextureThreshold(5)

        disparity_map = stereo.compute(left_gray, right_gray)

        return disparity_map

    def estimate_distance(self, depth_map):
        # Get mean depth averaged over center line of image
        mean_depth = np.mean(depth_map[:, depth_map.shape[1] // 2])

        
        return mean_depth
        


def main():
    rclpy.init()
    stereo_vision_node = StereoVisionNode()
    rclpy.spin(stereo_vision_node.node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
