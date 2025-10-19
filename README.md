# Traffic_analyzer
Task CV-2-13 for NSU "Project Introduction" course

## Task description
Analyze traffic flow by detecting motion, counting vehicles crossing a line, and detecting road markings using computer vision techniques.

## Guideline (Original Task Requirements)
1. Capture video of the road.
2. Detect motion (CV-2-11).
3. Count objects crossing the line.
4. Detect road markings (CV-2-37).
5. Output: 'Cars: N, Road markings intact.'

## Features
- **Video File Processing**: Analyzes video files for traffic analysis
- **Motion Detection**: Advanced motion detection with shadow removal and noise filtering
- **Object Tracking**: Multi-object tracking with unique ID assignment and crossing detection
- **Road Marking Detection**: Automated road marking detection using edge detection and Hough line transforms (CV-2-37)
- **Visual Feedback**: Real-time display with object tracking, counting line, and statistics
- **Comprehensive Statistics**: Detailed car counting and road marking status reporting
- **Dual Window Display**: Separate windows for main analysis and road marking detection

## Functions

### `TrafficAnalyzer(video_source, line_x=None)`
Main traffic analyzer class for processing video files and detecting traffic.

**Parameters:**
- `video_source` (str): Path to video file
- `line_x` (int): X coordinate of the counting line (None for auto-detection)

**Features:**
- Motion detection with shadow removal
- Object tracking and counting
- Road marking detection
- Real-time visual feedback

### `find_differences(image1, image2, threshold=30)`
Detects differences between two frames for motion detection.

**Parameters:**
- `image1` (numpy.ndarray): First frame for comparison
- `image2` (numpy.ndarray): Second frame for comparison
- `threshold` (int): Threshold for difference detection (default: 30)

**Returns:**
- `numpy.ndarray`: Binary mask showing detected motion areas

**Features:**
- Shadow detection and removal
- Noise filtering with morphological operations
- Gradient-based filtering for improved accuracy

### `road_marking_detection(frame, lowCannyThresh=150, highCannyThresh=200)`
Detects road markings in the frame using edge detection and line detection.

**Parameters:**
- `frame` (numpy.ndarray): Input frame for road marking detection
- `lowCannyThresh` (int): Lower threshold for Canny edge detection (default: 150)
- `highCannyThresh` (int): Upper threshold for Canny edge detection (default: 200)

**Returns:**
- `tuple`: (output_frame, line_count) - Frame with detected lines and count of detected lines

**Features:**
- Roberts filter for edge enhancement
- Canny edge detection for line detection
- Hough line transform for line segment detection
- Automatic road marking integrity assessment

### `detect_objects(motion_mask)`
Detects and tracks objects from motion detection mask.

**Parameters:**
- `motion_mask` (numpy.ndarray): Binary mask from motion detection

**Returns:**
- `list`: List of detected objects with tracking information

**Features:**
- Object tracking with unique IDs
- Motion-based object detection
- Automatic object lifecycle management

### `check_line_crossing(objects, frame_width)`
Checks if objects have crossed the counting line.

**Parameters:**
- `objects` (list): List of detected objects
- `frame_width` (int): Width of the frame for line positioning

**Features:**
- Vertical line crossing detection
- Prevents double-counting of objects
- Configurable crossing buffer zone

### `run(show_road_markings=True, show_motion=True)`
Main execution loop for real-time traffic analysis.

**Parameters:**
- `show_road_markings` (bool): Whether to show the road markings window (default: True)
- `show_motion` (bool): Whether to show the motion detection window (default: True)

**Features:**
- Real-time video processing
- Motion detection and object tracking
- Road marking analysis
- Configurable window display
- Visual feedback with statistics
- Interactive controls (press 'q' to quit)

## Running the Application

### Video File Analysis
```bash
python traffic_analyzer.py path/to/video.mp4
```

### Example with Test Video
```bash
python traffic_analyzer.py Test.mp4
```

### Usage
The application requires a video file path as a command line argument:
```bash
python traffic_analyzer.py <video_file_path>
```

## Dependencies
- `opencv-python>=4.5.0`: Computer vision operations
- `numpy>=1.19.0`: Numerical computations and array operations

## Installation
```bash
# Install Python dependencies
pip install opencv-python numpy
```

## Implementation Details

### Motion Detection (CV-2-11)
The implementation uses advanced motion detection techniques:
- **Frame Differencing**: Compares consecutive frames to detect changes
- **Shadow Removal**: Uses gradient analysis to filter out shadows
- **Noise Filtering**: Morphological operations to remove noise
- **Contour Analysis**: Area and aspect ratio filtering for object detection

### Road Marking Detection (CV-2-37)
The road marking detection system includes:
- **Roberts Filter**: Edge enhancement for better line detection
- **Canny Edge Detection**: Precise edge detection with configurable thresholds
- **Hough Line Transform**: Line segment detection and analysis
- **Integrity Assessment**: Automatic determination of road marking condition

### Object Tracking
The tracking system provides:
- **Unique ID Assignment**: Each object gets a persistent identifier
- **Motion Prediction**: Tracks object movement across frames
- **Crossing Detection**: Detects when objects cross the counting line
- **Lifecycle Management**: Automatic cleanup of lost objects

## Usage Examples

### Basic Traffic Analysis
```python
from traffic_analyzer import TrafficAnalyzer

# Create analyzer for video file
analyzer = TrafficAnalyzer(video_source="traffic_video.mp4")
analyzer.run()  # Shows both motion detection and road markings windows
```

### Custom Window Display
```python
# Show only road markings window
analyzer = TrafficAnalyzer(video_source="traffic_video.mp4")
analyzer.run(show_road_markings=True, show_motion=False)

# Show only motion detection window
analyzer = TrafficAnalyzer(video_source="traffic_video.mp4")
analyzer.run(show_road_markings=False, show_motion=True)

# Show main traffic analyzer only (no additional windows)
analyzer = TrafficAnalyzer(video_source="traffic_video.mp4")
analyzer.run(show_road_markings=False, show_motion=False)
```

### Custom Line Position
```python
# Set custom counting line position
analyzer = TrafficAnalyzer(video_source="traffic_video.mp4", line_x=400)
analyzer.run()
```

### Current Line Configuration
The system currently uses a vertical counting line positioned 60 pixels to the left of the frame center:
```python
# Default line position: (frame_width // 2)
analyzer = TrafficAnalyzer(video_source="traffic_video.mp4")  # Auto-detects line position
```

## Output Format
The system provides real-time output in the format:
```
Cars: N, Road markings intact.
```

## Visual Interface
- **Main Window**: Shows live video with object tracking and vertical counting line
- **Motion Detection Window**: Displays motion detection mask (optional)
- **Road Markings Window**: Displays detected road markings with green lines (optional)
- **Statistics Overlay**: Real-time car count and road marking status
- **Object Tracking**: Visual rectangles and IDs for tracked objects
- **Motion Detection**: Green rectangles around detected motion areas
- **Object Tracking**: Blue rectangles with unique IDs for tracked objects

### Window Control
The `run()` function now supports configurable window display:
- **`show_motion=True`**: Shows motion detection window
- **`show_road_markings=True`**: Shows road markings window
- Both parameters default to `True` for full functionality

## Controls
- **Press 'q'**: Quit the application
- **Press 'ESC'**: Alternative quit method

## Project Structure
```
Traffic_analyzer/
├── traffic_analyzer.py          # Main application
├── requirements.txt             # Dependencies
├── README.md                   # This documentation
├── Project-task-CV-2-37/        # Road marking detection
│   └── Road_detector.py        # Original CV-2-37 code
└── Test.mp4                    # Sample video file
```

## Materials
- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Computer Vision Tutorials](https://opencv-python-tutroals.readthedocs.io/)
- [Motion Detection Techniques](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_contours_begin.html)
- [Hough Line Transform](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html)
- [CV-2-37 Task Implementation](Project-task-CV-2-37/Road_detector.py)