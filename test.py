import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# 초기 프레임 읽기
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Optical Flow 계산
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.1, 0)

    # 움직임이 감지된 부분 추출
    magnitude_threshold = 10
    mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

    motion_mask = mag > magnitude_threshold
    # 감지된 부분이 있는지 확인
    if np.any(motion_mask):
        # 감지된 부분의 바운딩 박스 계산
        contours, _ = cv.findContours(motion_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bounding_box = cv.boundingRect(np.vstack(contours))

        # 바운딩 박스의 중심점 계산
        center_x = bounding_box[0] + bounding_box[2] // 2
        center_y = bounding_box[1] + bounding_box[3] // 2

        # 감지된 부분에 바운딩 박스와 중심점 표시
        frame_with_box = frame2.copy()
        cv.rectangle(frame_with_box, (bounding_box[0], bounding_box[1]),
                     (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 2)
        cv.circle(frame_with_box, (center_x, center_y), 5, (0, 0, 255), -1)

        # 중심점 좌표 값을 화면에 나타내기
        cv.putText(frame_with_box, f'Center: ({center_x}, {center_y})', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # 결과 화면에 표시
        cv.imshow('Motion Detection', frame_with_box)
    else:
        # 감지된 움직임이 없다는 메시지 표시
        cv.putText(frame2, 'No Motion Detected', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # 결과 화면에 표시
        cv.imshow('Motion Detection', frame2)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    prvs = next

cap.release()
cv.destroyAllWindows()
