import gi
import cv2
import numpy as np
import threading
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class GstreamerOpenCVProcessor:
    def __init__(self, rtsp_urls):
        Gst.init(None)
        
        self.frames = {}
        self.frame_locks = {}
        self.pipelines = []
        self.mainloop = GLib.MainLoop()
        self.thread = None
        self.running = False
        
        for i, url in enumerate(rtsp_urls):
            # Create lock for thread-safe frame access
            self.frame_locks[i] = threading.Lock()
            self.frames[i] = None
            
            # Create pipeline with hardware acceleration + OpenCV compatibility
            pipeline_str = (
                f"rtspsrc location={url} latency=0 ! "
                "rtph264depay ! h264parse ! "
                "nvv4l2decoder enable-max-performance=1 ! " # Hardware decoder
                "nvvideoconvert ! video/x-raw,format=BGRx ! " # Hardware converter
                "videoconvert ! video/x-raw,format=BGR ! " # Convert to BGR for OpenCV
                f"appsink name=sink{i} max-buffers=1 drop=true sync=false emit-signals=true"
            )
            
            pipeline = Gst.parse_launch(pipeline_str)
            self.pipelines.append(pipeline)
            
            # Get appsink for retrieving frames
            appsink = pipeline.get_by_name(f"sink{i}")
            appsink.connect("new-sample", self.on_new_sample, i)
    
    def on_new_sample(self, sink, stream_id):
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame info from caps
            structure = caps.get_structure(0)
            width = structure.get_value("width") 
            height = structure.get_value("height")
            
            # Convert Gst.Buffer to numpy array (OpenCV frame)
            success, map_info = buf.map(Gst.MapFlags.READ)
            if success:
                # Create numpy array from buffer data
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                # Make a copy since the buffer will be freed
                frame = frame.copy()
                
                # Update frame in thread-safe way
                with self.frame_locks[stream_id]:
                    self.frames[stream_id] = frame
                
                buf.unmap(map_info)
            
            return Gst.FlowReturn.OK
        
        return Gst.FlowReturn.ERROR
    
    def process_frames(self):
        while self.running:
            for stream_id in self.frames:
                # Get frame in thread-safe way
                with self.frame_locks[stream_id]:
                    frame = self.frames[stream_id]
                
                if frame is not None:
                    # Process frame with OpenCV
                    processed = self.process_frame(frame.copy(), stream_id)
                    
                    # You can store results, display, or perform further analysis
                    # For example, write to disk periodically:
                    # if self.frame_count % 30 == 0:
                    #     cv2.imwrite(f"processed_{stream_id}_{self.frame_count}.jpg", processed)
            
            # Add a small sleep to prevent high CPU usage
            GLib.usleep(1000)  # 1ms
    
    def process_frame(self, frame, stream_id):
        # Your OpenCV processing code
        
        # Example: Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Add your own processing here
        
        return frame
    
    def start(self):
        self.running = True
        
        # Start all pipelines
        for pipeline in self.pipelines:
            pipeline.set_state(Gst.State.PLAYING)
        
        # Start processing thread
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.daemon = True
        self.thread.start()
        
        # Run mainloop for GStreamer
        self.mainloop.run()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        
        for pipeline in self.pipelines:
            pipeline.set_state(Gst.State.NULL)
        
        self.mainloop.quit()

# Usage
rtsp_urls = [
    "rtsp://username:password@ip:port/stream1",
    "rtsp://username:password@ip:port/stream2"
]
processor = GstreamerOpenCVProcessor(rtsp_urls)
try:
    processor.start()
except KeyboardInterrupt:
    processor.stop()