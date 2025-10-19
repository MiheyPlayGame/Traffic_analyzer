import cv2
import numpy as np
import sys
import os
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque

# Import CV-2-37 road marking detection function
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Project-task-CV-2-37'))
from Road_detector import road_marking_detection as cv37_road_marking_detection


@dataclass
class DetectionConfig:
    """Configuration parameters for traffic detection"""
    # Motion detection parameters
    motion_threshold: int = 30
    min_contour_area: int = 100
    shadow_gradient_threshold: int = 20
    min_area_filter: int = 50
    max_aspect_ratio: float = 5.0
    
    # Object tracking parameters
    tracking_distance_threshold: int = 50
    max_frames_since_seen: int = 10
    crossing_buffer: int = 10
    
    # Road marking detection parameters
    canny_low_threshold: int = 150
    canny_high_threshold: int = 200
    hough_rho: float = 1.5
    hough_theta: float = np.pi/180
    hough_threshold: int = 50
    min_line_length: int = 20
    max_line_gap: int = 20
    road_marking_threshold: int = 3
    
    # Visual parameters
    line_thickness: int = 5
    font_scale: float = 0.7
    font_thickness: int = 2


class TrafficAnalyzer:
    """
    Advanced traffic analyzer with motion detection and road marking analysis.
    
    Combines CV-2-11 (motion detection) and CV-2-37 (road marking detection)
    for comprehensive traffic flow analysis.
    """
    
    def __init__(self, video_source: str, line_x: Optional[int] = None):
        """
        Initialize the Traffic Analyzer with configuration.
        
        Args:
            video_source: Path to video file
            line_x: X coordinate of the counting line (None for auto-detection)
            
        Raises:
            ValueError: If video source is invalid
            RuntimeError: If video capture initialization fails
        """
        self.config = DetectionConfig()
        self.video_source = video_source
        self.line_x = 900
        
        # Initialize video capture
        self._initialize_video_capture()
        
        # Traffic analysis state
        self.car_count = 0
        self.object_id = 0
        self.tracking_objects: Dict[int, Dict[str, Any]] = {}
        self.previous_frame: Optional[np.ndarray] = None
        self.road_markings_intact = True
        
        print(f"TrafficAnalyzer initialized with video source: {video_source}")
    
    def _initialize_video_capture(self) -> None:
        """Initialize video capture with error handling."""
        try:
            # Validate video file path
            if not self.video_source or not isinstance(self.video_source, str):
                raise ValueError("Video source must be a valid file path")
            
            if not os.path.exists(self.video_source):
                raise ValueError(f"Video file does not exist: {self.video_source}")
            
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.video_source}")
            
            # Test if we can read a frame
            ret, _ = self.cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read from video file: {self.video_source}")
            
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        except Exception as e:
            print(f"Error: Video capture initialization failed: {e}")
            raise
    
    
    def detect_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Detect motion between two frames using advanced techniques.
        
        Args:
            frame1: First frame for comparison
            frame2: Second frame for comparison
            
        Returns:
            Binary mask showing detected motion areas
            
        Raises:
            ValueError: If frames have incompatible dimensions
        """
        try:
            # Validate input frames
            if frame1.shape != frame2.shape:
                raise ValueError("Frames must have the same dimensions")
            
            # Convert to grayscale if needed
            gray1 = self._to_grayscale(frame1)
            gray2 = self._to_grayscale(frame2)
            
            # Calculate frame difference
            diff = cv2.absdiff(gray1, gray2)
            _, binary_mask = cv2.threshold(diff, self.config.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Apply noise filtering
            binary_mask = self._apply_noise_filtering(binary_mask)
            
            # Remove shadows using gradient analysis
            binary_mask = self._remove_shadows(binary_mask, gray1, gray2)
            
            # Filter by area and aspect ratio
            binary_mask = self._filter_by_contour_properties(binary_mask)
            
            return binary_mask
            
        except Exception as e:
            print(f"Error: Motion detection failed: {e}")
            return np.zeros_like(gray1, dtype=np.uint8)
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _apply_noise_filtering(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to remove noise."""
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        return binary_mask
    
    def _remove_shadows(self, binary_mask: np.ndarray, gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
        """Remove shadow regions using gradient analysis."""
        # Calculate gradient magnitude for both frames
        grad1 = self._calculate_gradient_magnitude(gray1)
        grad2 = self._calculate_gradient_magnitude(gray2)
        
        # Shadow regions have low gradient
        shadow_mask = (grad1 < self.config.shadow_gradient_threshold) & \
                     (grad2 < self.config.shadow_gradient_threshold)
        shadow_mask = shadow_mask.astype(np.uint8) * 255
        
        # Remove shadow regions from motion detection
        return cv2.bitwise_and(binary_mask, cv2.bitwise_not(shadow_mask))
    
    def _calculate_gradient_magnitude(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate gradient magnitude using Sobel operators."""
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def _filter_by_contour_properties(self, binary_mask: np.ndarray) -> np.ndarray:
        """Filter contours by area and aspect ratio."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(binary_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.min_area_filter:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                if aspect_ratio < self.config.max_aspect_ratio:
                    cv2.fillPoly(filtered_mask, [contour], 255)
        
        return filtered_mask
    
    def detect_road_markings(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect road markings using CV-2-37 algorithm.
        
        Uses the original CV-2-37 road_marking_detection function.
        
        Args:
            frame: Input frame for road marking detection
            
        Returns:
            Tuple of (output_frame_with_lines, line_count)
        """
        try:
            # Use the original CV-2-37 road marking detection function
            output_frame = cv37_road_marking_detection(
                frame, 
                self.config.canny_low_threshold, 
                self.config.canny_high_threshold
            )
            
            # Count the number of green lines (road markings) in the output
            line_count = self._count_road_markings(output_frame)
            
            # Update road marking integrity status based on line count
            self.road_markings_intact = line_count >= self.config.road_marking_threshold
            
            return output_frame, line_count
            
        except Exception as e:
            print(f"Error: Road marking detection failed: {e}")
            return frame.copy(), 0
    
    def _count_road_markings(self, frame: np.ndarray) -> int:
        """
        Count the number of road markings (green lines) in the frame.
        
        Args:
            frame: Frame with detected road markings
            
        Returns:
            Number of detected road marking lines
        """
        try:
            # Convert to HSV for better green color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define range for green color (road markings)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            
            # Create mask for green pixels
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours of green regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count significant green regions (road markings)
            line_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter out small noise
                    line_count += 1
            
            return line_count
            
        except Exception as e:
            print(f"Error: Road marking counting failed: {e}")
            return 0
    
    
    def detect_and_track_objects(self, motion_mask: np.ndarray) -> List[Tuple[int, int, int, int, int, int, int]]:
        """
        Detect and track objects from motion detection mask.
        
        Args:
            motion_mask: Binary mask from motion detection
            
        Returns:
            List of detected objects with tracking information
        """
        try:
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.config.min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w // 2, y + h // 2
                    
                    # Find closest existing object or create new one
                    obj_id = self._find_or_create_object(center_x, center_y, x, y, w, h)
                    objects.append((obj_id, x, y, w, h, center_x, center_y))
            
            # Clean up lost objects
            self._cleanup_lost_objects()
            
            return objects
            
        except Exception as e:
            print(f"Error: Object detection failed: {e}")
            return []
    
    def _find_or_create_object(self, center_x: int, center_y: int, x: int, y: int, w: int, h: int) -> int:
        """Find closest existing object or create new one."""
        closest_id = None
        min_distance = float('inf')
        
        for obj_id, obj_data in self.tracking_objects.items():
            last_center = obj_data['last_center']
            distance = np.sqrt((center_x - last_center[0])**2 + (center_y - last_center[1])**2)
            
            if distance < self.config.tracking_distance_threshold and distance < min_distance:
                min_distance = distance
                closest_id = obj_id
        
        if closest_id is not None:
            # Update existing object
            self.tracking_objects[closest_id].update({
                'last_center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'frames_since_seen': 0
            })
            return closest_id
        else:
            # Create new object
            self.object_id += 1
            self.tracking_objects[self.object_id] = {
                'last_center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'frames_since_seen': 0,
                'crossed_line': False
            }
            return self.object_id
    
    def _cleanup_lost_objects(self) -> None:
        """Remove objects that haven't been seen for too long."""
        to_remove = []
        for obj_id, obj_data in self.tracking_objects.items():
            obj_data['frames_since_seen'] += 1
            if obj_data['frames_since_seen'] > self.config.max_frames_since_seen:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracking_objects[obj_id]
    
    def check_line_crossing(self, objects: List[Tuple[int, int, int, int, int, int, int]], 
                          frame_width: int) -> None:
        """
        Check if objects have crossed the counting line.
        
        Args:
            objects: List of detected objects
            frame_width: Width of the frame for line positioning
        """
        if self.line_x is None:
            self.line_x = (frame_width // 2)
        
        for obj_id, x, y, w, h, center_x, center_y in objects:
            if obj_id in self.tracking_objects:
                obj_data = self.tracking_objects[obj_id]
                
                if not obj_data['crossed_line']:
                    if (self.line_x - self.config.crossing_buffer < center_x < 
                        self.line_x + self.config.crossing_buffer):
                        obj_data['crossed_line'] = True
                        self.car_count += 1
                        print(f"Car detected! Total count: {self.car_count}")
    
    def _draw_visual_elements(self, frame: np.ndarray, objects: List[Tuple[int, int, int, int, int, int, int]], 
                            motion_mask: np.ndarray, line_count: int) -> np.ndarray:
        """Draw all visual elements on the frame."""
        # Draw motion detection rectangles
        frame = self._draw_motion_rectangles(frame, motion_mask)
        
        # Draw object tracking rectangles
        frame = self._draw_object_rectangles(frame, objects)
        
        # Draw counting line
        frame = self._draw_counting_line(frame)
        
        # Draw information overlay
        frame = self._draw_information_overlay(frame, line_count)
        
        return frame
    
    def _draw_motion_rectangles(self, frame: np.ndarray, motion_mask: np.ndarray) -> np.ndarray:
        """Draw rectangles around detected motion areas."""
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > self.config.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
    
    def _draw_object_rectangles(self, frame: np.ndarray, objects: List[Tuple[int, int, int, int, int, int, int]]) -> np.ndarray:
        """Draw rectangles around tracked objects."""
        for obj_id, x, y, w, h, center_x, center_y in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return frame
    
    def _draw_counting_line(self, frame: np.ndarray) -> np.ndarray:
        """Draw the counting line."""
        frame_height = frame.shape[0]
        cv2.line(frame, (self.line_x, 0), (self.line_x, frame_height), (0, 0, 255), self.config.line_thickness)
        cv2.putText(frame, "Counting Line", (self.line_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, (0, 0, 255), self.config.font_thickness)
        return frame
    
    def _draw_information_overlay(self, frame: np.ndarray, line_count: int) -> np.ndarray:
        """Draw information overlay on the frame."""
        # Car count
        cv2.putText(frame, f"Cars: {self.car_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Road marking status
        road_status = "Road markings intact" if self.road_markings_intact else "Road markings damaged"
        color = (0, 255, 0) if self.road_markings_intact else (0, 0, 255)
        cv2.putText(frame, road_status, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, color, self.config.font_thickness)
        
        # Line count
        cv2.putText(frame, f"Lines detected: {line_count}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, show_road_markings: bool = True, show_motion: bool = True) -> None:
        """
        Main execution loop for real-time traffic analysis.
        
        Args:
            show_road_markings: Whether to show the road markings window
            show_motion: Whether to show the motion detection window

        Raises:
            RuntimeError: If video processing fails
        """
        try:
            print("Traffic Analyzer started. Press 'q' to quit.")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Warning: Failed to read frame")
                    break
                
                frame_height, frame_width = frame.shape[:2]
                
                # Initialize line position if not set
                if self.line_x is None:
                    self.line_x = (frame_width // 2)
                
                # Motion detection and object tracking
                objects = []
                motion_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                
                if self.previous_frame is not None:
                    motion_mask = self.detect_motion(self.previous_frame, frame)
                    objects = self.detect_and_track_objects(motion_mask)
                    self.check_line_crossing(objects, frame_width)
                
                # Road marking detection
                road_marking_frame, line_count = self.detect_road_markings(frame)
                
                # Draw visual elements
                frame = self._draw_visual_elements(frame, objects, motion_mask, line_count)
                
                # Display frames
                if show_motion:
                    cv2.imshow('Motion Detection', frame)
                if show_road_markings:
                    cv2.imshow('Road Markings', road_marking_frame)
                
                # Update previous frame
                self.previous_frame = frame.copy()
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Final output
            road_status = "Road markings intact" if self.road_markings_intact else "Road markings damaged"
            print(f"\nFinal Results:")
            print(f"Cars: {self.car_count}, {road_status}.")
            
        except Exception as e:
            print(f"Error: Traffic analysis failed: {e}")
            raise RuntimeError(f"Traffic analysis failed: {e}")
        
        finally:
            self._cleanup()
    
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            print("Resources cleaned up successfully")
        except Exception as e:
            print(f"Error: Cleanup failed: {e}")


def main() -> None:
    """
    Main function to run the traffic analyzer.
    
    Requires video file path as command line argument.
    """
    try:
        # Parse command line arguments
        if len(sys.argv) < 2:
            print("Usage: python traffic_analyzer.py <video_file_path>")
            print("Example: python traffic_analyzer.py Test.mp4")
            sys.exit(1)
        
        video_source = sys.argv[1]
        print(f"Analyzing video file: {video_source}")
        analyzer = TrafficAnalyzer(video_source=video_source)
        analyzer.run()
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()