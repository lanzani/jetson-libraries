import cv2
import subprocess


def check_plugin_available(plugin_name, debug=False):
    """Check if a GStreamer plugin is available."""
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", plugin_name],
            capture_output=True,
            text=True,
            timeout=2,
        )
        # Check both stdout and stderr for error messages
        output = result.stdout + result.stderr
        is_available = (
            result.returncode == 0
            and "No such element" not in output
            and "No such plugin" not in output
            and "ERROR" not in output.upper()
        )
        if debug:
            print(f"  Checking {plugin_name}: returncode={result.returncode}, available={is_available}")
        return is_available
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        if debug:
            print(f"  Checking {plugin_name}: Exception - {e}")
        return False


def build_pipeline(unparsed_source, target_shape, debug=False, force_nvidia=False):
    """Build GStreamer pipeline with automatic plugin detection."""
    # Try NVIDIA GPU-accelerated plugins first
    # nvv4l2decoder/nvvideoconvert are from DeepStream SDK (preferred)
    # nvh264dec/nvh265dec/nvdec are from gst-plugins-bad nvcodec plugin (built with gst-build)
    # Priority: nvv4l2decoder first (DeepStream, best performance), then nvh264dec (nvcodec)
    nvidia_decoders = ["nvv4l2decoder", "nvh264dec", "nvh265dec", "nvdec"]
    nvidia_converters = ["nvvideoconvert", "nvvidconv"]

    decoder = None
    converter = None

    # Try NVIDIA plugins first (we know they're available from gst-inspect-1.0)
    # If force_nvidia is True or if checking fails, use NVIDIA plugins directly
    if force_nvidia:
        decoder = nvidia_decoders[0]  # nvh264dec (from nvcodec plugin)
        converter = nvidia_converters[0] if nvidia_converters else None  # nvvideoconvert (from DeepStream)
        print(f"Force using NVIDIA decoder: {decoder}")
        if converter:
            print(f"Force using NVIDIA converter: {converter}")
    else:
        if debug:
            print("Checking NVIDIA decoders...")
        # Find available NVIDIA decoder
        for nv_dec in nvidia_decoders:
            if check_plugin_available(nv_dec, debug=debug):
                decoder = nv_dec
                print(f"Using NVIDIA decoder: {decoder}")
                break

        if debug:
            print("Checking NVIDIA converters...")
        # Find available NVIDIA converter
        for nv_conv in nvidia_converters:
            if check_plugin_available(nv_conv, debug=debug):
                converter = nv_conv
                print(f"Using NVIDIA converter: {converter}")
                break

        # Don't use NVIDIA plugins if check failed - they're not actually available

    # Build pipeline based on available plugins
    if decoder and converter:
        # Full NVIDIA GPU-accelerated pipeline
        if decoder == "nvv4l2decoder":
            decoder_str = f"{decoder} enable-max-performance=1"
        else:
            decoder_str = decoder

        source = (
            f"rtspsrc location={unparsed_source} latency=200 ! "
            f"queue ! rtph264depay ! h264parse ! "
            f"{decoder_str} ! "
            f"{converter} ! video/x-raw,format=BGRx,width={target_shape[1]},height={target_shape[0]} ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"queue ! appsink max-buffers=60 drop=true sync=false"
        )
        print("Using GPU-accelerated NVIDIA pipeline")
    elif decoder:
        # We have NVIDIA decoder but no converter - use standard converter
        # nvdec/nvh264dec output NV12 format, need conversion
        if decoder == "nvv4l2decoder":
            decoder_str = f"{decoder} enable-max-performance=1"
        else:
            decoder_str = decoder

        source = (
            f"rtspsrc location={unparsed_source} latency=200 ! "
            f"queue ! rtph264depay ! h264parse ! "
            f"{decoder_str} ! "
            f"videoscale ! video/x-raw,width={target_shape[1]},height={target_shape[0]} ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"queue ! appsink max-buffers=60 drop=true sync=false"
        )
        print(f"Using GPU-accelerated NVIDIA decoder ({decoder}) with standard converter")
    else:
        # Fallback to standard GStreamer plugins (CPU-based)
        # Try hardware-accelerated decoders first, then software
        hw_decoders = ["vaapih264dec", "nvh264dec"]
        sw_decoder = "avdec_h264"

        decoder = None
        for hw_dec in hw_decoders:
            if check_plugin_available(hw_dec):
                decoder = hw_dec
                print(f"Using hardware decoder: {decoder}")
                break

        if not decoder:
            decoder = sw_decoder
            print(f"Using software decoder: {decoder}")

        source = (
            f"rtspsrc location={unparsed_source} latency=200 ! "
            f"queue ! rtph264depay ! h264parse ! "
            f"{decoder} ! "
            f"videoscale ! video/x-raw,width={target_shape[1]},height={target_shape[0]} ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"queue ! appsink max-buffers=60 drop=true sync=false"
        )
        print("Using standard GStreamer pipeline (CPU-based)")

    return source


def main():
    import sys

    unparsed_source = "rtsp://admin:password12.@10.0.1.92/Preview_01_main"
    target_shape = (360, 640)

    # Enable debug mode if --debug flag is passed
    debug = "--debug" in sys.argv
    # Force NVIDIA plugins if --force-nvidia flag is passed (skip plugin checking)
    force_nvidia = "--force-nvidia" in sys.argv

    source = build_pipeline(unparsed_source, target_shape, debug=debug, force_nvidia=force_nvidia)
    print(f"\nPipeline: {source}\n")

    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: VideoCapture not opened")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
