# -*- coding: utf-8 -*-

import time
import cv2 
from flask import Flask, render_template, Response
from video_detector import detect_violation, video_detect
import winsound

app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('dashboard.html')


def gen():
    """Video streaming generator function. 768x576.avi"""
    cap = cv2.VideoCapture('videotester.mp4')

    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret:
            cap = cv2.VideoCapture('videotester.mp4')
            continue
        if ret:  # if there is a frame continue with code
            frame = detect_violation(frame)
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

        key = cv2.waitKey(20)
        if key == 27:
            break

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    

if __name__ == '__main__':
    app.run(debug=False)