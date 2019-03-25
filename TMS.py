import cv2
import numpy as np

# Used in live steaming.
#cap = cv2.VideoCapture("rtsp://93.157.18.93/media/video1") #SNC-CH110 - sony network camera

cap = cv2.VideoCapture("test/vid/h264.mp4") #in case video stream is inaccessible

####################### Background extraction. Using accumulateWeighted.
alpha = 0.01
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #src
avg = np.float32(gray) #dst
while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.accumulateWeighted(gray, avg, alpha)
    res_avg = cv2.convertScaleAbs(avg)
    cv2.imwrite("test/img/average.jpg", res_avg)
    cv2.imshow("accumulateWeighted", res_avg)
    cv2.imshow("input", frame)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


####################### Template creation using mask.
mask_files = ['test/img/mask_roi_1.jpg',
              'test/img/mask_roi_2.jpg',
              'test/img/mask_roi_3.jpg',
              'test/img/mask_roi_4.jpg']

template_files = ['test/img/template_roi_1.jpg',
                  'test/img/template_roi_2.jpg',
                  'test/img/template_roi_3.jpg',
                  'test/img/template_roi_4.jpg']

crop = [{"x1": 970, "y1": 400, "x2":1279, "y2":500},
        {"x1": 755, "y1": 275, "x2":940, "y2":365},
        {"x1": 335, "y1": 335, "x2":528, "y2":400},
        {"x1": 300, "y1": 470, "x2":670, "y2":675}]

masks = [cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) for mask_file in mask_files]

average = cv2.imread("test/img/average.jpg", cv2.IMREAD_GRAYSCALE)


for index, (name, mask) in enumerate(zip(template_files, masks)):
    template = cv2.bitwise_and(average, average, mask=mask)

    x1 = crop[index]["x1"]
    y1 = crop[index]["y1"]
    x2 = crop[index]["x2"]
    y2 = crop[index]["y2"]

    template = template[y1:y2, x1:x2]
    cv2.imwrite(name, template)
    cv2.imshow(name, template)

cv2.waitKey(0)
cv2.destroyAllWindows()


####################### ROI extraction.

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while ret:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("input", frame)

    for index, (name, mask) in enumerate(zip(mask_files, masks)):
        res = cv2.bitwise_and(gray, gray, mask=mask)
        x1 = crop[index]["x1"]
        y1 = crop[index]["y1"]
        x2 = crop[index]["x2"]
        y2 = crop[index]["y2"]


        res = res[y1:y2, x1:x2]
        cv2.imwrite("roi_{}.jpg".format(index+1), res)

        cv2.imshow(name, res)

    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


####################### Density counting.

templates = [cv2.imread(template_file, cv2.IMREAD_GRAYSCALE) for template_file in template_files]

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


while ret:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, (mask, template) in enumerate(zip(masks, templates)):
        roi = cv2.bitwise_and(gray, gray, mask=mask)
        x1 = crop[index]["x1"]
        y1 = crop[index]["y1"]
        x2 = crop[index]["x2"]
        y2 = crop[index]["y2"]


        roi = roi[y1:y2, x1:x2]
        res = cv2.matchTemplate(roi, template, cv2.TM_CCORR_NORMED)
        cv2.putText(roi,"{:.3f}".format(res[0][0]),(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)

#         cv2.imshow("Template {}".format(index), template)
        cv2.imshow("Roi {}".format(index), roi)


    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
