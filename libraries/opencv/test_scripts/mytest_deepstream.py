import sys
import gi
import numpy as np
import cv2
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from gi.repository import GstApp

class DeepStreamOpenCVProcessor:
    def __init__(self, rtsp_urls):
        # Initialize GStreamer
        Gst.init(None)
        
        self.number_of_streams = len(rtsp_urls)
        self.mainloop = GLib.MainLoop()
        self.pipeline = self.build_pipeline(rtsp_urls)
        self.appsinks = []
        
        # Get appsink for each stream
        for i in range(self.number_of_streams):
            sink = self.pipeline.get_by_name(f"appsink{i}")
            sink.set_property("emit-signals", True)
            sink.connect("new-sample", self.on_new_sample, i)
            self.appsinks.append(sink)
    
    def build_pipeline(self, rtsp_urls):
        # Create streammux for batching inputs
        pipeline_str = "nvstreammux name=mux batch-size=4 width=1280 height=720 ! nvvideoconvert ! "
        
        # Add tee to fork streams for display and processing
        pipeline_str += "tee name=t "
        
        # Add each stream source
        for i, url in enumerate(rtsp_urls):
            src_str = f"rtspsrc location={url} latency=0 ! rtph264depay ! h264parse ! "
            src_str += "nvv4l2decoder enable-max-performance=1 ! nvvideoconvert ! "
            src_str += f"video/x-raw,format=BGRx ! queue ! mux.sink_{i} "
            pipeline_str = src_str + pipeline_str
        
        # Add branch for processing with appsink for each stream
        for i in range(len(rtsp_urls)):
            pipeline_str += f"t. ! nvstreammux name=mux_out{i} batch-size=1 width=1280 height=720 ! "
            pipeline_str += f"nvvideoconvert ! video/x-raw(memory:NVMM),format=BGRx ! "
            pipeline_str += f"nvvideoconvert ! video/x-raw,format=BGR ! "
            pipeline_str += f"appsink name=appsink{i} max-buffers=1 drop=true sync=false "
        
        return Gst.parse_launch(pipeline_str)
    
    def on_new_sample(self, appsink, stream_id):
        sample = appsink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame dimensions from caps
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")
            
            # Map buffer to numpy array (OpenCV format)
            success, map_info = buf.map(Gst.MapFlags.READ)
            if success:
                # Create numpy array from buffer data
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                
                # Process frame with OpenCV
                processed_frame = self.process_frame(frame, stream_id)
                
                # Unmap buffer
                buf.unmap(map_info)
            
            return Gst.FlowReturn.OK
        
        return Gst.FlowReturn.ERROR
    
    def process_frame(self, frame, stream_id):
        # Your OpenCV processing here
        # Example: grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Example: edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Do your processing here...
        
        # You can save the frame, display it, or perform further analysis
        # cv2.imwrite(f"processed_frame_{stream_id}.jpg", edges)
        
        return edges
    
    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.mainloop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.mainloop.quit()

# Usage
rtsp_urls = [
    "rtsp://username:password@ip:port/stream1",
    "rtsp://username:password@ip:port/stream2"
]
processor = DeepStreamOpenCVProcessor(rtsp_urls)
processor.start()