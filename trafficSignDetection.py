import cv2
import numpy as np

def process_video(path = "./Videos/MVI_1049.avi"):
	cap = cv2.VideoCapture(path)

	while (cap.isOpened()):
		# Take each frame
		ret, frame = cap.read()

		# Use Gaussian Blur to reduce high frequency noise
		# and allow us to focus on the structural objects inside the frame
		blurred = cv2.GaussianBlur(frame, (3, 3), 0)
		#blurred = cv2.medianBlur(frame, 5)
		
		# Convert BGR to HSV
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		
		# Threshold the HSV image to get only red
		red1 = cv2.inRange(hsv, (0, 100, 100), (15, 255, 255))
		red2 = cv2.inRange(hsv, (160, 100, 120), (180, 255, 255))
		red_mask = cv2.add(red1, red2)
		#red_mask = cv2.dilate(red_mask)

		# Threshold the HSV image to get only blue
		blue_mask = cv2.inRange(hsv, (100, 120, 100), (120, 255, 255))

		mask = cv2.add(red_mask, blue_mask)

		mask = cv2.erode(mask, None, iterations = 1)
		mask = cv2.dilate(mask, None, iterations = 3)

		# Find contours in the mask
		im, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Proceed if at least one contour was found
		if len(cnts) > 0:
			#cv2.drawContours(frame, cnts, -1, (0, 255, 0), 1)

			for each in cnts:
				#(xCenter, yCenter), radius = cv2.minEnclosingCircle(each)
				#center = (int(xCenter), int(yCentqer))
				#radius = int(radius)

				
				#cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
				cv2.drawContours(image = mask, contours = each, contourIdx = -1, color = 255, thickness = -1)
				mask = cv2.dilate(mask, None, iterations = 5)
				mask = cv2.erode(mask, None, iterations = 5)

					#cv2.circle(frame, center, radius, (0, 0, 255), 1)

		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		for each in cnts:
			x, y, w, h = cv2.boundingRect(each)
			if w > 10 and h > 10 and h/w > 0.9 and float(h)/w < 1.5:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
		#result = cv2.bitwise_and(frame, frame, mask)
		
		cv2.imshow("frame", frame)
		#cv2.imshow("hsv", hsv)
		cv2.imshow("mask", mask)
		#cv2.imshow("result", result)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

process_video("./Videos/MVI_1049.avi")
process_video("./Videos/MVI_1054.avi")
