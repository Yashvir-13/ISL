import os 
import cv2
data_dir='Data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
num_classes=35
dataset_size=200
cap=cv2.VideoCapture(0)
for j in range(10,19+1):
    if not os.path.exists(os.path.join(data_dir,str(j))):
        os.makedirs(os.path.join(data_dir,str(j)))
    print("Collecting data for class{}".format(j))
    done=False
    while True:
        ret,frame=cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()