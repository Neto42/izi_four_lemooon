import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MULT', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[6]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.legacy.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.legacy.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create()
        if tracker_type == 'MULT':
            trackers = cv2.legacy.MultiTracker_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        if tracker_type == 'CSRT':
            tracker = cv2.legacy.TrackerCSRT_create()

    video = cv2.VideoCapture("frame/video/car2.mp4")
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    cv2.putText(frame, 'exit to ESC', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(frame, 'start to ENter', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    bbox = cv2.selectROI(frame, True)
    try:
        ok = tracker.init(frame, bbox)
    except Exception as e:
        e

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()

        ok, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if ok:
            for i in range(length):
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                image = cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                img_copy = image.copy()
                cv2.rectangle(img_copy, p1, p2, (255, 255, 255), 2)
                cv2.imshow("Image", img_copy)

                sub_img = image[p1[1]:p2[1], p1[0]:p2[0]]
                cv2.imwrite(f'frame/frame/frame{i}.jpg', sub_img)

        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27: break
