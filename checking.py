import cv2
import numpy as np
import os

def test_video_codecs():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    codecs_to_test = [('H264', 'mp4'), ('XVID', 'avi'), ('MJPG', 'avi')]
    
    for codec_name, extension in codecs_to_test:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            test_file = f"test_{codec_name}.{extension}"
            out = cv2.VideoWriter(test_file, fourcc, 30.0, (640, 480))
            
            if out.isOpened():
                out.write(frame)
                out.release()
                if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
                    print(f"✅ {codec_name} works with .{extension}")
                    os.remove(test_file)
                else:
                    print(f"❌ {codec_name} failed")
            else:
                print(f"❌ {codec_name} failed")
        except Exception as e:
            print(f"❌ {codec_name} failed: {e}")

test_video_codecs()