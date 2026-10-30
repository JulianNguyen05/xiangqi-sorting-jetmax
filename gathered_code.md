# GATHERED SOURCE CODE - XIANGQI SORTING JETMAX

- **Thư mục gốc:** `D:\2026_2027\Chuyen-de\xiangqi-sorting-jetmax`
- **Tổng số file đã gôm:** 13

## Mục lục

### Nhóm 1: API Robot (Quan trọng nhất)
- [docs\robot-reference\jetmax_control\scripts\jetmax_control_main.py](#docsrobot-referencejetmaxcontrolscriptsjetmaxcontrolmainpy)
- [docs\robot-reference\jetmax_control\msg\JetMax.msg](#docsrobot-referencejetmaxcontrolmsgjetmaxmsg)
- [docs\robot-reference\jetmax_control\msg\SetJetMax.msg](#docsrobot-referencejetmaxcontrolmsgsetjetmaxmsg)
- [docs\robot-reference\jetmax_control\msg\SetJoint.msg](#docsrobot-referencejetmaxcontrolmsgsetjointmsg)
- [docs\robot-reference\jetmax_control\msg\Sucker.msg](#docsrobot-referencejetmaxcontrolmsgsuckermsg)
- [docs\robot-reference\jetmax_control\msg\SuckerState.msg](#docsrobot-referencejetmaxcontrolmsgsuckerstatemsg)
- [docs\robot-reference\waste_classification_stepper.py](#docsrobot-referencewasteclassificationstepperpy)

### Nhóm 2: Code cũ có thể tái sử dụng
- [robot\motion.py](#robotmotionpy)
- [robot\aruco_locator.py](#robotarucolocatorpy)
- [robot\rig.py](#robotrigpy)
- [tests\pick_place.py](#testspickplacepy)
- [tests\joint_teleop.py](#testsjointteleoppy)

### Nhóm 3: Thông tin Model & Training
- [training\README.md](#trainingreadmemd)

---

## Nhóm 1: API Robot (Quan trọng nhất)

### `docs\robot-reference\jetmax_control\scripts\jetmax_control_main.py`

*Dung lượng: 8.67 KB | Số dòng: 182*

```python
#!/usr/bin/env python3
import actionlib
import rospy
from jetmax_control.msg import Mecanum
from jetmax_control.msg import JetMax as JetMaxState
from jetmax_control.msg import SetServo, SetJoint, SetJetMax
from std_msgs.msg import UInt8, UInt16, Float32, Bool
from std_srvs.srv import Empty
import hiwonder
import threading
import action_set


# import jetmax_control_joystick as joystick


class JetMaxControl:
    def __init__(self):
        rospy.init_node('jetmax_control', anonymous=True)
        self.pwm_servos = [hiwonder.pwm_servo1, hiwonder.pwm_servo2]

        """
        麦轮相关的topic
        """
        try:
            self.chassis = hiwonder.MecanumChassis()
            self.mecanum_pub = rospy.Publisher('/jetmax/mecanum/status', Mecanum, queue_size=1)
            self.mecanum_sub = rospy.Subscriber('/jetmax/mecanum/command', Mecanum,
                                                queue_size=1,
                                                callback=lambda msg: self.chassis.set_velocity(msg.velocity,
                                                                                               msg.direction,
                                                                                               msg.angular_rate))
        except Exception:
            self.chassis = None
            self.mecanum_sub = None
            self.mecanum_pub = None

        """
        机械臂相关的topic
        """
        self.jetmax = hiwonder.JetMax()
        # 机械臂末端位置控制topic
        self.jetmax_pub = rospy.Publisher('/jetmax/status', JetMaxState, queue_size=1)
        self.jetmax_sub = rospy.Subscriber('/jetmax/command', SetJetMax,
                                           queue_size=1,
                                           callback=lambda msg: self.jetmax.set_position((msg.x, msg.y, msg.z),
                                                                                         msg.duration))
        # 机械臂末端带速度的位置控制 topic
        self.jetmax_speed_sub = rospy.Subscriber('/jetmax/speed_command', SetJetMax,
                                                 queue_size=1,
                                                 callback=lambda msg: self.jetmax.set_position_with_speed(
                                                     (msg.x, msg.y, msg.z), max(msg.duration, 0.001)))
        # 机械臂末端相对位置控制topic
        self.jetmax_relative_sub = rospy.Subscriber('/jetmax/relative_command', SetJetMax,
                                                    queue_size=1,
                                                    callback=lambda msg: self.jetmax.set_position_relatively(
                                                        (msg.x, msg.y, msg.z), msg.duration))

        self.jetmax_joint_pubs = []
        self.jetmax_joint_subs = []
        self.jetmax_servo_pubs = []
        self.jetmax_servo_subs = []
        self.jetmax_servo_speed_subs = []
        # 机械臂各个关节的topic
        for i in range(1, 4):
            joint_pub = rospy.Publisher('/jetmax/joint{}/status'.format(i), Float32, queue_size=1)
            joint_cmd = rospy.Subscriber('/jetmax/joint{}/command'.format(i), SetJoint,
                                         queue_size=1,
                                         callback_args=i,
                                         callback=lambda msg, joint_id: self.jetmax.set_joint(joint_id,
                                                                                              msg.data,
                                                                                              msg.duration))
            servo_pub = rospy.Publisher('/jetmax/servo{}/status'.format(i), UInt16, queue_size=1)
            servo_cmd = rospy.Subscriber('/jetmax/servo{}/command'.format(i), SetServo,
                                         queue_size=1,
                                         callback_args=i,
                                         callback=lambda msg, servo_id: self.jetmax.set_servo(servo_id,
                                                                                              msg.data,
                                                                                              msg.duration))
            servo_speed_cmd = rospy.Subscriber('/jetmax/servo{}/speed_command'.format(i), SetServo,
                                               queue_size=1,
                                               callback_args=i,
                                               callback=lambda msg, servo_id: self.jetmax.set_servo_with_speed(
                                                   servo_id,
                                                   msg.data,
                                                   msg.duration))
            self.jetmax_joint_pubs.append(joint_pub)
            self.jetmax_joint_subs.append(joint_cmd)
            self.jetmax_servo_pubs.append(servo_pub)
            self.jetmax_servo_subs.append(servo_cmd)
            self.jetmax_servo_speed_subs.append(servo_speed_cmd)

        """
        末端执行器的topic
        """
        self.end_effector_servo_pubs = []
        self.end_effector_servo_subs = []
        # 末端的几个小的topic
        for i in range(1, 3):
            pub = rospy.Publisher('/jetmax/end_effector/servo{}/status'.format(i), UInt8, queue_size=1)
            cmd = rospy.Subscriber('/jetmax/end_effector/servo{}/command'.format(i), SetServo,
                                   queue_size=1,
                                   callback_args=i - 1,
                                   callback=lambda msg, servo_id: self.pwm_servos[servo_id].set_position(msg.data,
                                                                                                         msg.duration))
            self.end_effector_servo_pubs.append(pub)
            self.end_effector_servo_subs.append(cmd)

        self.sucker = hiwonder.Sucker()
        # 末端吸嘴的topic
        self.sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/status', Bool, queue_size=1)
        self.sucker_sub = rospy.Subscriber('/jetmax/end_effector/sucker/command', Bool,
                                           queue_size=1,
                                           callback=lambda msg: self.sucker.set_state(msg.data))

        # 回初始位置的topic
        self.go_home_srv = rospy.Service('/jetmax/go_home', Empty, self.go_home)
        self.action_set = action_set.ActionSetRunner(self.jetmax, self.sucker)

        self.rate = rospy.Rate(10)  # 10hz
        self.go_home() # 让机械臂回初始位置

    def go_home(self, req=None):
        self.jetmax.go_home(2)
        return []

    def update(self):
        while not rospy.is_shutdown():
            """
            更新麦轮底盘的状态
            """
            if self.chassis:
                mecanum_status = Mecanum(
                    velocity=self.chassis.velocity,
                    direction=self.chassis.direction,
                    angular_rate=self.chassis.angular_rate
                )
                self.mecanum_pub.publish(mecanum_status)

            """
            更新末端执行器的状态
            """
            pwm_servo_states = []
            for i in range(2):
                servo_state = UInt8(data=int(self.pwm_servos[i].get_position()))
                self.end_effector_servo_pubs[i].publish(servo_state)
                pwm_servo_states.append(servo_state)
            self.sucker_pub.publish(self.sucker.get_state())

            """
            更新机械臂的状态
            """
            servo_states = []
            joint_states = []
            for i in range(3):
                servo_state = int(self.jetmax.servos[i])
                joint_state = self.jetmax.joints[i]
                self.jetmax_servo_pubs[i].publish(UInt16(data=servo_state))
                self.jetmax_joint_pubs[i].publish(Float32(data=joint_state))
                servo_states.append(servo_state)
                joint_states.append(joint_state)
            x, y, z = self.jetmax.position
            self.jetmax_pub.publish(JetMaxState(
                x=x, y=y, z=z,
                joint1=joint_states[0], joint2=joint_states[1], joint3=joint_states[2],
                servo1=servo_states[0], servo2=servo_states[1], servo3=servo_states[2],
                pwm1=hiwonder.pwm_servo1.get_position(), pwm2=hiwonder.pwm_servo2.get_position(),
                sucker=self.sucker.get_state()
            ))

            self.rate.sleep()


if __name__ == '__main__':
    jetmax = JetMaxControl()
    # joystick.jetmax = jetmax.jetmax
    # joystick.sucker = jetmax.sucker
    try:
        #    threading.Thread(target=joystick.joystick_loop, daemon=True).start()
        jetmax.update()
    except rospy.ROSInterruptException:
        pass
```

---

### `docs\robot-reference\jetmax_control\msg\JetMax.msg`

*Dung lượng: 161 B | Số dòng: 12*

```text
float32 x
float32 y
float32 z
float32 joint1
float32 joint2
float32 joint3
float32 servo1
float32 servo2
float32 servo3
float32 pwm1
float32 pwm2
bool    sucker
```

---

### `docs\robot-reference\jetmax_control\msg\SetJetMax.msg`

*Dung lượng: 46 B | Số dòng: 4*

```text
float32 x
float32 y
float32 z
float32 duration
```

---

### `docs\robot-reference\jetmax_control\msg\SetJoint.msg`

*Dung lượng: 29 B | Số dòng: 2*

```text
float32 data
float32 duration
```

---

### `docs\robot-reference\jetmax_control\msg\Sucker.msg`

*Dung lượng: 10 B | Số dòng: 1*

```text
bool state
```

---

### `docs\robot-reference\jetmax_control\msg\SuckerState.msg`

*Dung lượng: 9 B | Số dòng: 1*

```text
bool data
```

---

### `docs\robot-reference\waste_classification_stepper.py`

*Dung lượng: 10.41 KB | Số dòng: 283*

```python
#!/usr/bin/env python3
import os
import sys
import cv2
import math
import time
import rospy
import numpy as np
import threading
import queue
import hiwonder
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from jetmax_control.msg import SetServo
from yolov5_tensorrt import Yolov5TensorRT

ROS_NODE_NAME = "waste_classification_stepper"

TRT_ENGINE_PATH = os.path.join(sys.path[0], "models/waste_v5_160.trt")
TRT_INPUT_SIZE = 160
TRT_CLASS_NAMES = ('Banana Peel', 'Broken Bones', 'Cigarette End', 'Disposable Chopsticks',
                   'Ketchup', 'Marker', 'Oral Liquid Bottle', 'Plate',
                   'Plastic Bottle', 'Storage Battery', 'Toothbrush', 'Umbrella')
TRT_NUM_CLASSES = 12
WASTE_CLASSES = {
    'food_waste': ('Banana Peel', 'Broken Bones', 'Ketchup'),
    'hazardous_waste': ('Marker', 'Oral Liquid Bottle', 'Storage Battery'),
    'recyclable_waste': ('Plastic Bottle', 'Toothbrush', 'Umbrella'),
    'residual_waste': ('Plate', 'Cigarette End', 'Disposable Chopsticks'),
}
COLORS = {
    'recyclable_waste': (0, 0, 255),
    'hazardous_waste': (255, 0, 0),
    'food_waste': (0, 255, 0),
    'residual_waste': (80, 80, 80)
}

TARGET_POSITION = {
    'recyclable_waste': (0, -180, 5, 5200),
    'hazardous_waste': (0, -180, 5, 4400),
    'food_waste': (0, -180, 5, 3600),
    'residual_waste': (0, -180, 5, 2800)
}

# 现实坐标转换函数
def camera_to_world(cam_mtx, r, t, img_points):
    inv_k = np.asmatrix(cam_mtx).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(r, r_mat)
    # invR * T
    inv_r = np.asmatrix(r_mat).I  # 3*3
    transPlaneToCam = np.dot(inv_r, np.asmatrix(t))  # 3*3 dot 3*1 = 3*1
    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0][0]
        coords[1][0] = img_pt[0][1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)  # 3*3 dot 3*1 = 3*1
        # [x,y,1] * invR
        worldPtPlane = np.dot(inv_r, worldPtCam)  # 3*3 dot 3*1 = 3*1
        # zc
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        # zc * [x,y,1] * invR
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        # [X,Y,Z]=zc*[x,y,1]*invR - invR*T
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)  # 3*1 dot 1*3 = 3*3
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt


class WasteClassification:
    def __init__(self):
        self.lock = threading.RLock()
        self.is_running = False
        self.moving_box = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.camera_params = None
        self.fps_t0 = time.time()
        self.fps = 0
        self.K = None
        self.R = None
        self.T = None

    def reset(self):
        self.is_running = False
        self.moving_box = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0

    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/card_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)

# 移动控制函数
def moving():
    try:
        c_x, c_y, waste_class_name = state.moving_box # 目标的画面坐标
        x, y, _ = camera_to_world(state.K, state.R, state.T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0] #转换成现实距离
        cur_x, cur_y, cur_z = jetmax.position  # 当前的机械臂位置
        t = math.sqrt(x * x + y * y + 140 * 140) / 140 #计算移动的时间
        new_x, new_y = cur_x + x, cur_y + y #目标的现实坐标
        arm_angle = math.atan(new_y / new_x) * 180 / math.pi # 计算偏转角
        if arm_angle > 0:
            arm_angle = (90 - arm_angle)
        elif arm_angle < 0:
            arm_angle = (-90 - arm_angle)
        else:
            pass
        hiwonder.pwm_servo1.set_position(90 - arm_angle, 0.1) #角度补偿

        x, y, z, stepper_position = TARGET_POSITION[waste_class_name] # 读取放置坐标
        jetmax.set_position((new_x, new_y, 50), t) # 移动到目标上方
        rospy.sleep(t + 0.2)

        sucker.set_state(True) # 打开气泵
        jetmax.set_position((new_x, new_y, -12), 0.8) # 吸取目标
        rospy.sleep(0.8)
        jetmax.set_position((new_x, new_y, 30), 0.3)
        rospy.sleep(0.3)
        t = abs(new_x)/120.0 + abs(new_y-cur_y)/120.0
        hiwonder.pwm_servo1.set_position(90, 0.5) #角度回正
        jetmax.set_position((x, y, 50), t) # 移动到放置坐标上方
        rospy.sleep(t)

        stepper.set_mode(stepper.SLEEP) # 设置滑轨使能
        stepper.goto(stepper_position) # 驱动滑轨移动到放置位置
        stepper_time = stepper_position/1000 # 计算需要的时间
        rospy.sleep(stepper_time)

        jetmax.set_position((x, y, z), 0.5) # 到放置位置
        rospy.sleep(0.5)

        sucker.release(3)
        jetmax.set_position((x, y, z + 50), 0.5) 
        rospy.sleep(1)

    finally:
        jetmax.set_position((cur_x, cur_y, cur_z), 1) # 机械臂回到初始位置
        stepper.goto(-stepper_position) # 滑轨回到初始位置
        stepper_time = stepper_position/1000
        rospy.sleep(stepper_time)
        stepper.set_mode(stepper.EN) # 解除滑轨锁定
        state.moving_box = None
        state.runner = None
        print("FINISHED")

# 图像处理函数
def image_proc(image):
    if state.runner is not None:
        return image

    outputs = yolov5.detect(image) # 模型推理
    boxes, confs, classes = yolov5.post_process(image, outputs, 0.5)
    width = image.shape[1]
    height = image.shape[0]
    cards = []
    waste_name = 'None'
    for box, cls_conf, cls_id in zip(boxes, confs, classes):
        x1 = int(box[0] / TRT_INPUT_SIZE * width)
        y1 = int(box[1] / TRT_INPUT_SIZE * height)
        x2 = int(box[2] / TRT_INPUT_SIZE * width)
        y2 = int(box[3] / TRT_INPUT_SIZE * height)
        waste_name = TRT_CLASS_NAMES[cls_id]  # 得到检测到的分类名称
        
        waste_class_name = ''
        for k, v in WASTE_CLASSES.items():
            if waste_name in v:
                waste_class_name = k
                break
        cards.append((cls_conf, x1, y1, x2, y2, waste_class_name))
        cv2.putText(image, waste_class_name, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[waste_class_name], 2)
        cv2.putText(image, waste_name + "{:0.2f}".format(float(cls_conf)), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[waste_class_name], 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[waste_class_name], 3)

    if len(cards) == 0:
        state.count = 0
        state.moving_box = None
    else:
        if state.moving_box is None:
            moving_box = max(cards, key=lambda card: card[0])
            conf, x1, y1, x2, y2, waste_class_name = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            state.moving_box = c_x, c_y, waste_class_name
        else:
            l_c_x, l_c_y, l_waste_class_name = state.moving_box
            moving_box = min(cards, key=lambda card: math.sqrt((l_c_x - card[1]) ** 2 + (l_c_y - card[2]) ** 2))

            conf, x1, y1, x2, y2, waste_class_name = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            # 在画面中标记出来
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 6)
            image = cv2.circle(image, (int(c_x), int(c_y)), 1, (255, 255, 255), 10)

            if math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) > 30:
                state.count = 0
            else:
                c_x = l_c_x * 0.2 + c_x * 0.8
                c_y = l_c_y * 0.2 + c_y * 0.8
                state.count += 1
            state.moving_box = c_x, c_y, waste_class_name
            if state.count > 50: #多次检测
                state.count = 0
                print(waste_name,waste_class_name)
                state.runner = threading.Thread(target=moving, daemon=True) #开启移动子线程
                state.runner.start()

    return image


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


def image_proc_b():
    ros_image = image_queue.get(block=True)
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    result_img = image_proc(image)
    # fps cal
    fps_t1 = time.time()
    fps_cur = (1.0 / (fps_t1 - state.fps_t0))
    state.fps = fps_cur if state.fps == 0.0 else (state.fps * 0.8 + fps_cur * 0.2)
    state.fps_t0 = fps_t1
    show_fps(result_img, state.fps)
    image_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imshow(ROS_NODE_NAME, image_bgr)
    cv2.waitKey(1)


def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    image_queue = queue.Queue(maxsize=2)
    state = WasteClassification()
    state.load_camera_params()
    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()
    stepper = hiwonder.Stepper(1)
    jetmax.go_home(2,2)
    stepper.go_home(True)
    stepper.set_mode(stepper.EN)
    stepper.set_div(stepper.DIV_1_4)
    hiwonder.pwm_servo1.set_position(90, 0.5)
    rospy.sleep(1)
    yolov5 = Yolov5TensorRT(TRT_ENGINE_PATH, TRT_INPUT_SIZE, TRT_NUM_CLASSES)
    image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback) # 订阅摄像头画面

    while True:
        try:
            image_proc_b()
            if rospy.is_shutdown():
                sys.exit(0)
        except KeyboardInterrupt:
            break

```

---

## Nhóm 2: Code cũ có thể tái sử dụng

### `robot\motion.py`

*Dung lượng: 5.94 KB | Số dòng: 134*

```python
#!/usr/bin/env python3
# Điều khiển tay máy (ROS topic) + thanh trượt (hiwonder.Stepper) + đầu hút. Tách từ tests/pick_place.py để dùng chung.
# Cần node jetmax_control đang chạy và rospy.init_node() đã được gọi trước khi tạo Mover.
# KHÔNG tạo hiwonder.JetMax() ở đây vì node jetmax_control đã giữ bus servo.

import math, time
import rospy
from std_msgs.msg import Bool
from jetmax_control.msg import SetJetMax, JetMax

X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0   # giống các script trước
TOL = 5.0                       # mm, sai lệch cho phép giữa đích và vị trí thực
FAST, SLOW = 100.0, 40.0        # mm/giây: di chuyển bình thường / khi hạ xuống chạm quân cờ
SLIDER_MIN, SLIDER_MAX = 0, 5200
SLIDER_SPEED = 800.0            # bước/giây, dùng ước lượng thời gian chờ (hãng ghi 1000; để thấp cho chắc)
SLIDER_MIN_Z = 40.0             # chỉ trượt khi tay máy nâng đủ cao


def in_limits(p):
    x, y, z = p
    return (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN)


class Mover:
    def __init__(self, use_slider=True):
        self.cmd = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
        self.sucker = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
        self.stepper, self.pos, self.enabled = None, 0, False
        if use_slider:
            import hiwonder
            self.stepper = hiwonder.Stepper(1)
        rospy.sleep(1.0)
        self.start = self.status()
        self.sucker.publish(Bool(data=False))

    # ---------------------------------------------------------------- tay máy
    def status(self):
        m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
        return m.x, m.y, m.z

    def goto(self, p, speed=FAST):
        if not in_limits(p):
            raise ValueError('Điểm (%.1f, %.1f, %.1f) ngoài giới hạn an toàn' % tuple(p))
        cur = self.status()
        dist = math.sqrt(sum((p[i] - cur[i]) ** 2 for i in range(3)))
        t = max(0.5, dist / speed)
        self.cmd.publish(SetJetMax(x=p[0], y=p[1], z=p[2], duration=t))
        rospy.sleep(t + 0.3)
        now = self.status()
        err = max(abs(now[i] - p[i]) for i in range(3))
        if err > TOL:
            raise RuntimeError('Tay máy chưa tới đích (lệch %.1f mm): lệnh có thể bị chặn bởi giới hạn' % err)

    # --------------------------------------------------------------- thanh trượt
    def home_slider(self):
        if self.stepper is None:
            return
        if self.status()[2] < SLIDER_MIN_Z:
            raise RuntimeError('Tay máy đang thấp hơn z=%d, hãy nâng lên trước' % SLIDER_MIN_Z)
        st = self.stepper
        st.go_home(True)                         # chặn cho tới khi về home, không có timeout, canh nút nguồn
        st.set_mode(st.EN)
        st.set_div(st.DIV_1_4)
        self.pos, self.enabled = 0, False

    def slide_to(self, target):
        if self.stepper is None:
            return
        target = int(max(SLIDER_MIN, min(SLIDER_MAX, target)))
        d = target - self.pos
        if d == 0:
            return
        if self.status()[2] < SLIDER_MIN_Z:
            raise RuntimeError('Tay máy đang thấp hơn z=%d, không trượt' % SLIDER_MIN_Z)
        if not self.enabled:
            self.stepper.set_mode(self.stepper.SLEEP)   # chỉ gửi một lần
            self.enabled = True
        print('  thanh trượt %d -> %d' % (self.pos, target))
        self.stepper.goto(d)                     # bước TƯƠNG ĐỐI
        time.sleep(abs(d) / SLIDER_SPEED + 0.3)
        self.pos = target

    # ------------------------------------------------------------ thao tác cấp cao
    # sq_pos = (slider, x, y) lấy từ Board.position()
    def hover(self, sq_pos, zsafe):
        s, x, y = sq_pos
        self.slide_to(s)
        self.goto((x, y, zsafe))

    def pick(self, sq_pos, zpick, zsafe):
        self.hover(sq_pos, zsafe)
        _, x, y = sq_pos
        self.sucker.publish(Bool(data=True))
        rospy.sleep(0.3)
        self.goto((x, y, zpick), SLOW)
        rospy.sleep(0.8)                         # chờ hút bám chắc
        self.goto((x, y, zsafe), SLOW)

    def place(self, sq_pos, zplace, zsafe):
        self.hover(sq_pos, zsafe)
        _, x, y = sq_pos
        self.goto((x, y, zplace), SLOW)
        self.sucker.publish(Bool(data=False))    # thả, mở van xả khí
        rospy.sleep(1.0)
        self.goto((x, y, zsafe), SLOW)

    # ------------------------------------------------------------------ an toàn
    def safe_stop(self, zsafe):
        """Tắt hút và nâng tay máy lên z an toàn (dùng khi có lỗi giữa chừng)."""
        self.sucker.publish(Bool(data=False))
        try:
            cur = self.status()
            if cur[2] < zsafe:
                self.cmd.publish(SetJetMax(x=cur[0], y=cur[1], z=zsafe, duration=1.0))
                rospy.sleep(1.4)
        except Exception as e:
            print('Không nâng được tay máy (%s), hãy kiểm tra thủ công' % e)

    def restore(self, zsafe=50.0):
        """Về trạng thái ban đầu: tắt hút, nâng lên, thanh trượt về 0, tay máy về vị trí lúc bắt đầu."""
        self.sucker.publish(Bool(data=False))
        rospy.sleep(0.6)
        try:
            if self.stepper is not None and self.pos != 0:
                self.safe_stop(max(zsafe, SLIDER_MIN_Z))
                self.slide_to(0)
            self.cmd.publish(SetJetMax(x=self.start[0], y=self.start[1], z=self.start[2], duration=2.0))
            rospy.sleep(2.5)
        except Exception as e:
            print('Không tự về được vị trí ban đầu (%s), hãy kiểm tra thủ công' % e)
        if self.stepper is not None:
            self.stepper.set_mode(self.stepper.EN)
        print('Đã về vị trí ban đầu')
```

---

### `robot\aruco_locator.py`

*Dung lượng: 6.97 KB | Số dòng: 152*

```python
#!/usr/bin/env python3
# Định vị tờ bàn cờ A4 trong ảnh camera bằng 4 marker ArUco ở góc giấy.
# Từ các góc marker nhìn thấy (toạ độ mm đã biết trong configs/board_a4.yaml) tính phép biến đổi phối cảnh
# ảnh <-> mm của mặt bàn cờ, rồi chiếu 90 giao điểm lên ảnh. Nhờ đó KHÔNG cần bấm chuột đọc pixel để điền views.yaml.
#
# Cần OpenCV có module aruco (opencv-contrib). Kiểm tra:  python3 -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
# Tương thích cả API cũ (OpenCV < 4.7: DetectorParameters_create) lẫn mới (ArucoDetector).

import numpy as np

from board import square_name, NF, NR


def _make_detect(dict_name):
    import cv2
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('OpenCV (%s) không có module aruco. Cài opencv-contrib-python (hoặc bản OpenCV có contrib) '
                           'hoặc dùng cách bấm tay trong views.yaml.' % cv2.__version__)
    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        raise RuntimeError('OpenCV không có từ điển %s' % dict_name)
    d = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    if hasattr(aruco, 'ArucoDetector'):                       # OpenCV >= 4.7
        p = aruco.DetectorParameters()
        p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        det = aruco.ArucoDetector(d, p)
        return lambda gray: det.detectMarkers(gray)[:2]
    p = aruco.DetectorParameters_create()                     # OpenCV cũ (4.1 - 4.6)
    p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return lambda gray: aruco.detectMarkers(gray, d, parameters=p)[:2]


class Fix(object):
    """Phép biến đổi ảnh <-> mm của mặt bàn cờ tính từ các marker nhìn thấy trong MỘT ảnh."""

    def __init__(self, h_img2mm, ids, rms_mm, corners_px):
        self.H = h_img2mm
        self.Hinv = np.linalg.inv(h_img2mm)
        self.ids = sorted(ids)
        self.rms_mm = rms_mm              # sai số khớp của các góc marker (mm); 0 nếu chỉ có 1 marker (không đo được)
        self.corners_px = corners_px      # {id: mảng 4x2 pixel}

    def to_mm(self, x, y):
        v = self.H.dot(np.array([x, y, 1.0]))
        return v[0] / v[2], v[1] / v[2]

    def to_px(self, mx, my):
        """Điểm (mx, my) mm trong hệ BOARD -> pixel. Cũng nhận một tuple (mx, my) làm đối số duy nhất."""
        if my is None:
            mx, my = mx
        v = self.Hinv.dot(np.array([mx, my, 1.0]))
        return v[0] / v[2], v[1] / v[2]

    def px_per_mm(self, mx, my):
        """Tỉ lệ pixel/mm quanh điểm (mx, my) (trung bình theo hai trục)."""
        p0 = np.array(self.to_px(mx, my))
        px = np.array(self.to_px(mx + 10.0, my)) - p0
        py = np.array(self.to_px(mx, my + 10.0)) - p0
        return float((np.hypot(*px) + np.hypot(*py)) / 20.0)


class Locator(object):
    def __init__(self, geom):
        self.geom = geom
        self._detect = _make_detect(geom.dictionary)

    def markers(self, bgr):
        """{id: mảng 4x2 pixel} của các marker thuộc tờ bàn cờ nhìn thấy trong ảnh."""
        import cv2
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        corners, ids = self._detect(gray)
        out = {}
        if ids is None:
            return out
        for c, i in zip(corners, ids.reshape(-1)):
            if int(i) in self.geom.centers:
                out[int(i)] = np.asarray(c, dtype=np.float64).reshape(4, 2)
        return out

    def __call__(self, bgr):
        """Trả về Fix, hoặc None nếu không thấy marker nào."""
        import cv2
        found = self.markers(bgr)
        if not found:
            return None
        src, dst = [], []
        for mid, c in found.items():
            for p, q in zip(c, self.geom.marker_corners_mm(mid)):
                src.append(p)
                dst.append(q)
        src, dst = np.array(src, np.float64), np.array(dst, np.float64)
        h, _ = cv2.findHomography(src, dst, 0)
        if h is None:
            return None
        proj = np.hstack([src, np.ones((len(src), 1))]).dot(h.T)
        proj = proj[:, :2] / proj[:, 2:3]
        rms = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1)))) if len(found) > 1 else 0.0
        return Fix(h, found.keys(), rms, found)


# ------------------------------------------------------------------ điểm dạy cho views.yaml -------
def visible_squares(fix, geom, size, piece_half, edge=0.03):
    """{(f, r): (x_px, y_px)} của các giao điểm nằm gọn trong ảnh (cả quân cờ), cùng điều kiện View.covers()."""
    w, h = float(size[0]), float(size[1])
    mx, my = edge * w + piece_half, edge * h + piece_half
    out = {}
    for f in range(NF):
        for r in range(NR):
            x, y = fix.to_px(*geom.square_mm(f, r))
            if mx <= x <= w - mx and my <= y <= h - my:
                out[(f, r)] = (x, y)
    return out


def points_for_view(fix, geom, size, piece_half, edge=0.03, max_points=16):
    """Chọn <= max_points giao điểm nhìn thấy, rải đều (lấy điểm xa nhau nhất lần lượt), dạng {'a0': [x, y]} cho views.yaml."""
    vis = visible_squares(fix, geom, size, piece_half, edge)
    if len(vis) < 4:
        return {}
    keys = sorted(vis)
    pts = np.array([vis[k] for k in keys])
    chosen = [int(np.argmin(pts[:, 0] + pts[:, 1]))]
    dist = np.hypot(*(pts - pts[chosen[0]]).T)
    while len(chosen) < min(max_points, len(keys)):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.hypot(*(pts - pts[i]).T))
    return dict((square_name(*keys[i]), [round(float(pts[i][0]), 1), round(float(pts[i][1]), 1)]) for i in sorted(chosen))


def draw_overlay(bgr, fix, geom, piece_half, edge=0.03):
    """Vẽ lưới 90 giao điểm đã chiếu (xanh = nằm gọn trong ảnh, xám = không) và các marker tìm thấy lên bản sao của ảnh."""
    import cv2
    img = bgr.copy()
    h, w = img.shape[:2]
    vis = visible_squares(fix, geom, (w, h), piece_half, edge)
    for f in range(NF):
        for r in range(NR):
            x, y = fix.to_px(*geom.square_mm(f, r))
            if not (0 <= x < w and 0 <= y < h):
                continue
            ok = (f, r) in vis
            cv2.circle(img, (int(x), int(y)), 3, (0, 200, 0) if ok else (150, 150, 150), -1)
            if f in (0, 4, 8) and r in (0, 5, 9):
                cv2.putText(img, square_name(f, r), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 0) if ok else (180, 180, 180), 1)
    for mid, c in fix.corners_px.items():
        cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, (0, 140, 255), 2)
        cv2.putText(img, 'id%d' % mid, (int(c[0][0]), int(c[0][1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
    cv2.putText(img, 'marker %s  sai so %.2f mm  %d/90 o' % (fix.ids, fix.rms_mm, len(vis)), (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img
```

---

### `robot\rig.py`

*Dung lượng: 3.78 KB | Số dòng: 80*

```python
#!/usr/bin/env python3
# HardwareRig: phần dính với robot thật, dùng chung cho scan_board.py, play_robot.py (và các công cụ sau này).
# Gồm: nạp mô hình, node ROS, camera, Mover (tay máy + thanh trượt + đầu hút), di chuyển tới một view, về vị trí chờ.
# Chỉ chạy trên robot (cần rospy, hiwonder, node jetmax_control và usb_cam); trên laptop dùng FakeWorld / --images.
# Import nặng (rospy, cv2, motion) đều làm trong __init__ nên import file này trên laptop không lỗi.

import time


class HardwareRig(object):
    TOPIC = '/usb_cam/image_rect_color'

    def __init__(self, model=None, conf=0.5, iou=0.45, z_safe=50.0, node='rig', log=print, ask=None, home=True):
        """model: đường dẫn best.pt/.onnx/.engine (None = chỉ chụp ảnh, không nhận diện).
        ask(text): hỏi người dùng trước các thao tác cần chú ý (nếu None thì không hỏi).
        home: đưa thanh trượt về HOME ngay khi khởi tạo."""
        import os
        import numpy as np
        import cv2
        import rospy
        from sensor_msgs.msg import Image
        import motion
        self.np, self.cv2, self.rospy, self.motion, self.log = np, cv2, rospy, motion, log
        self.detector = None
        if model:
            from detect_view import load_backend
            self.detector = load_backend(os.path.expanduser(model), conf, iou)        # nạp mô hình trước khi động tới robot
        rospy.init_node(node, anonymous=True)
        self._state = {'img': None, 'n': 0}
        rospy.Subscriber(self.TOPIC, Image, self._on_image, queue_size=1)
        self.mv = motion.Mover()
        self.lift = max(float(z_safe), motion.SLIDER_MIN_Z + 10.0)
        self.cur_view = '-'
        self.saver = None                      # hàm saver(tên_view, ảnh_bgr) gọi sau mỗi lần chụp, nếu có
        self.start = self.mv.start
        log('Vị trí ban đầu của tay máy: %s' % (self.start,))
        if home:
            if ask:
                ask('Đưa thanh trượt về HOME (tay máy phải đang nâng cao, không vướng)')
            self.mv.home_slider()

    # ---------------------------------------------------------------- camera
    def _on_image(self, msg):
        self._state['img'] = self.np.frombuffer(msg.data, self.np.uint8).reshape(msg.height, msg.width, 3).copy()   # RGB
        self._state['n'] += 1

    def grab(self):
        """Ảnh BGR mới nhất, đợi tới khi có khung hình mới sau khi tay máy đã dừng hẳn."""
        t0, n0 = time.time(), self._state['n']
        while self._state['n'] < n0 + 2:
            if time.time() - t0 > 3:
                raise RuntimeError('Không nhận được ảnh từ ' + self.TOPIC)
            time.sleep(0.02)
        bgr = self.cv2.cvtColor(self._state['img'], self.cv2.COLOR_RGB2BGR)       # mô hình train trên ảnh BGR
        if self.saver:
            self.saver(self.cur_view, bgr)
        return bgr

    # ---------------------------------------------------------------- di chuyển
    def raise_arm(self):
        p = self.mv.status()
        if p[2] < self.lift:
            self.mv.goto((p[0], p[1], self.lift))

    def move_to(self, view):
        """Đưa camera tới một view (có thuộc tính name, slider, arm). Chỉ trượt khi tay máy đã nâng đủ cao."""
        self.raise_arm()
        self.mv.slide_to(view.slider)
        self.mv.goto(tuple(view.arm))
        self.rospy.sleep(0.3)
        self.cur_view = view.name

    def park(self):
        """Về vị trí chờ (thanh trượt về 0, tay máy về chỗ lúc bắt đầu), tránh vướng tay người."""
        self.raise_arm()
        self.mv.slide_to(0)
        self.mv.goto(self.start)

    def cleanup(self, z_safe=50.0):
        self.mv.restore(z_safe)
```

---

### `tests\pick_place.py`

*Dung lượng: 8.18 KB | Số dòng: 172*

```python
#!/usr/bin/env python3
# Gắp ở điểm A, đặt ở điểm B bằng đầu hút, CÓ dùng thanh trượt (chưa dùng camera).
# Tay máy điều khiển qua ROS topic (cần node jetmax_control đang chạy), thanh trượt qua hiwonder.Stepper.
#
# Bước 1 (chỉ bay trên A và B, KHÔNG hạ xuống, KHÔNG hút; thanh trượt vẫn chạy thật):
#   python3 pick_place.py --a 0 -210.4 --b 0 -200.4 --zpick -8.8 --zplace -5.4 --sa 1500 --sb 5000 --hover
# Bước 2 (gắp - đặt thật, hỏi Enter trước mỗi pha):
#   python3 pick_place.py --a 0 -210.4 --b 0 -200.4 --zpick -8.8 --zplace -5.4 --sa 1500 --sb 5000
# Khi chắc chắn, thêm --yes để chạy liền không hỏi.
#
# Tọa độ A, B và số bước trượt lấy từ joint_teleop.py (bấm p). Số bước trượt là tính từ HOME của thanh trượt,
# nên script sẽ đưa thanh trượt về HOME trước khi bắt đầu (bỏ qua bằng --no-home nếu chắc chắn nó đang ở home).

import argparse, math, time
import rospy
from std_msgs.msg import Bool
from jetmax_control.msg import SetJetMax, JetMax

X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0   # giống các script trước
TOL = 5.0                      # mm, sai lệch cho phép giữa đích và vị trí thực
FAST, SLOW = 100.0, 40.0       # mm/giây: di chuyển bình thường / khi hạ xuống chạm quân cờ
SLIDER_MIN, SLIDER_MAX = 0, 5200
SLIDER_SPEED = 800.0           # bước/giây, dùng để ước lượng thời gian chờ (hãng ghi 1000; để thấp cho chắc)
SLIDER_MIN_Z = 40.0            # chỉ trượt khi tay máy nâng đủ cao

S = {'stepper': None, 'pos': 0, 'enabled': False}


def status():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return m.x, m.y, m.z


def check(p):
    x, y, z = p
    if not (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN):
        raise ValueError('Điểm (%.1f, %.1f, %.1f) ngoài giới hạn an toàn' % p)


def goto(pub, p, speed=FAST):
    check(p)
    cur = status()
    dist = math.sqrt(sum((p[i] - cur[i]) ** 2 for i in range(3)))
    t = max(0.5, dist / speed)
    pub.publish(SetJetMax(x=p[0], y=p[1], z=p[2], duration=t))
    rospy.sleep(t + 0.3)
    now = status()
    err = max(abs(now[i] - p[i]) for i in range(3))
    if err > TOL:
        raise RuntimeError('Tay máy chưa tới đích (lệch %.1f mm): lệnh có thể bị chặn bởi giới hạn' % err)


def slide_to(target):
    st = S['stepper']
    if st is None:
        raise RuntimeError('Không có thanh trượt')
    target = int(max(SLIDER_MIN, min(SLIDER_MAX, target)))
    d = target - S['pos']
    if d == 0:
        return
    if status()[2] < SLIDER_MIN_Z:
        raise RuntimeError('Tay máy đang thấp hơn z=%d, không trượt' % SLIDER_MIN_Z)
    if not S['enabled']:
        st.set_mode(st.SLEEP)                    # chỉ gửi một lần
        S['enabled'] = True
    print('  thanh trượt %d -> %d' % (S['pos'], target))
    st.goto(d)
    time.sleep(abs(d) / SLIDER_SPEED + 0.3)
    S['pos'] = target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, nargs=2, required=True, metavar=('X', 'Y'), help='điểm gắp')
    ap.add_argument('--b', type=float, nargs=2, required=True, metavar=('X', 'Y'), help='điểm đặt')
    ap.add_argument('--zpick', type=float, required=True, help='z khi đầu hút chạm quân cờ ở A')
    ap.add_argument('--zplace', type=float, default=None, help='z khi thả ở B (mặc định = zpick + 3)')
    ap.add_argument('--zsafe', type=float, default=50.0, help='z khi di chuyển ngang')
    ap.add_argument('--sa', type=int, default=0, help='vị trí thanh trượt khi gắp ở A (bước từ home)')
    ap.add_argument('--sb', type=int, default=0, help='vị trí thanh trượt khi đặt ở B (bước từ home)')
    ap.add_argument('--no-home', action='store_true', help='không đưa thanh trượt về home lúc bắt đầu')
    ap.add_argument('--hover', action='store_true', help='chỉ bay trên A và B, không hạ, không hút')
    ap.add_argument('--yes', action='store_true', help='không hỏi Enter trước mỗi pha')
    a = ap.parse_args()
    zplace = a.zplace if a.zplace is not None else a.zpick + 3
    A, B = a.a, a.b
    for s in (a.sa, a.sb):
        if not SLIDER_MIN <= s <= SLIDER_MAX:
            raise SystemExit('Vị trí thanh trượt %d ngoài khoảng %d..%d' % (s, SLIDER_MIN, SLIDER_MAX))

    def ask(text):
        if not a.yes and input(text + ' [Enter = chạy, q = dừng] ').strip().lower() == 'q':
            raise KeyboardInterrupt

    rospy.init_node('pick_place')
    abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    sucker = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
    need_slider = a.sa != 0 or a.sb != 0
    if need_slider:
        try:
            import hiwonder
            S['stepper'] = hiwonder.Stepper(1)
        except Exception as e:
            raise SystemExit('Không khởi tạo được thanh trượt: %s' % e)
    rospy.sleep(1.0)

    start = status()
    print('Vị trí ban đầu:', start)
    print('A=%s (trượt %d)  B=%s (trượt %d)  zpick=%.1f zplace=%.1f zsafe=%.1f %s' % (
        A, a.sa, B, a.sb, a.zpick, zplace, a.zsafe, '[CHỈ BAY, KHÔNG HẠ]' if a.hover else ''))
    for p in [(A[0], A[1], a.zsafe), (B[0], B[1], a.zsafe), (A[0], A[1], a.zpick), (B[0], B[1], zplace)]:
        check(p)                                 # kiểm tra mọi điểm trước khi di chuyển
    try:
        sucker.publish(Bool(data=False))
        if need_slider and not a.no_home:
            ask('Đưa thanh trượt về HOME (vị trí 0). Tay máy phải đang nâng cao và không vướng')
            if start[2] < SLIDER_MIN_Z:
                raise RuntimeError('Tay máy đang thấp hơn z=%d, hãy nâng lên trước' % SLIDER_MIN_Z)
            st = S['stepper']
            st.go_home(True)                     # chặn cho tới khi về home, không có timeout
            st.set_mode(st.EN)
            st.set_div(st.DIV_1_4)
            S['pos'], S['enabled'] = 0, False
        ask('Trượt tới %d và bay tới trên A' % a.sa)
        slide_to(a.sa)
        goto(abs_pub, (A[0], A[1], a.zsafe))
        if a.hover:
            ask('Trượt tới %d và bay tới trên B' % a.sb)
            slide_to(a.sb)
            goto(abs_pub, (B[0], B[1], a.zsafe))
        else:
            ask('Hạ xuống A và hút')
            sucker.publish(Bool(data=True))
            rospy.sleep(0.3)
            goto(abs_pub, (A[0], A[1], a.zpick), SLOW)
            rospy.sleep(0.8)                     # chờ hút bám chắc
            goto(abs_pub, (A[0], A[1], a.zsafe), SLOW)
            ask('Mang tới B (trượt tới %d)' % a.sb)
            slide_to(a.sb)
            goto(abs_pub, (B[0], B[1], a.zsafe))
            ask('Hạ xuống B và thả')
            goto(abs_pub, (B[0], B[1], zplace), SLOW)
            sucker.publish(Bool(data=False))     # thả, mở van xả khí
            rospy.sleep(1.0)
            goto(abs_pub, (B[0], B[1], a.zsafe), SLOW)
        print('Xong')
    except KeyboardInterrupt:
        print('Dừng theo yêu cầu')
    except (ValueError, RuntimeError) as e:
        print('LỖI:', e)
    finally:
        sucker.publish(Bool(data=False))
        rospy.sleep(0.6)
        try:
            cur = status()
            if S['stepper'] is not None and S['pos'] != 0 and cur[2] < SLIDER_MIN_Z:
                abs_pub.publish(SetJetMax(x=cur[0], y=cur[1], z=a.zsafe, duration=1.0))   # nâng lên trước khi trượt về
                rospy.sleep(1.4)
            if S['stepper'] is not None and S['pos'] != 0:
                slide_to(0)
            abs_pub.publish(SetJetMax(x=start[0], y=start[1], z=start[2], duration=2.0))
            rospy.sleep(2.5)
        except Exception as e:
            print('Không tự về được vị trí ban đầu (%s), hãy kiểm tra thủ công' % e)
        if S['stepper'] is not None:
            S['stepper'].set_mode(S['stepper'].EN)
        print('Đã về vị trí ban đầu')


if __name__ == '__main__':
    main()
```

---

### `tests\joint_teleop.py`

*Dung lượng: 9.60 KB | Số dòng: 256*

```python
#!/usr/bin/env python3
# Điều khiển JetMax bằng phím: tịnh tiến x/y/z, xoay TỪNG KHỚP, servo cổ tay, đầu hút và THANH TRƯỢT.
# Tay máy điều khiển qua ROS topic (cần node jetmax_control đang chạy); thanh trượt qua hiwonder.Stepper.
#
#   python3 joint_teleop.py            # thoát: tay máy và thanh trượt về chỗ ban đầu
#   python3 joint_teleop.py --stay     # thoát: để nguyên tại chỗ
#
# Muốn đi XA hơn: giữ Shift (gõ chữ HOA) = bước x4, hoặc bấm t để đổi mức bước.
# Tốc độ mỗi lần bấm tự tỉ lệ theo quãng đường nên bước nhỏ không còn chờ lâu.

import sys, time, select, termios, tty, argparse
import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from jetmax_control.msg import SetJetMax, SetJoint, SetServo, JetMax

MM_STEPS = [5, 10, 20, 40]               # mm
DEG_STEPS = [2, 5, 10, 20]               # độ (khớp và cổ tay)
SLIDER_STEPS = [500, 1000, 2000, 3000]   # bước thanh trượt (bước nhỏ làm động cơ dừng-chạy liên tục nên giật)
BIG = 4                                  # hệ số khi giữ Shift
SPEED_MM = 150.0                         # mm/giây
SPEED_DEG = 120.0                        # độ/giây
MIN_DUR = 0.2                            # giây, thời gian tối thiểu mỗi lệnh
X_LIM, Y_LIM, Z_LIM = (-120, 120), (-230, -100), (-12, 225)   # giống các script trước
R_MIN = 50.0                             # bán kính xy tối thiểu (jetmax.py bỏ qua lệnh nếu nhỏ hơn)
PWM_MIN, PWM_MAX = 0, 180
JOINT_TOL = 0.5                          # độ
SLIDER_MIN, SLIDER_MAX = 0, 5200         # 5200 là điểm xa nhất trong ví dụ hãng, chưa phải hành trình đo được
SLIDER_SPEED = 800.0                     # bước/giây dùng để ước lượng thời gian chờ (hãng ghi 1000; để thấp hơn cho chắc)
SLIDER_MIN_Z = 40.0                      # chỉ cho trượt khi tay máy nâng đủ cao (ví dụ hãng nâng lên z=50 trước)

ap = argparse.ArgumentParser()
ap.add_argument('--stay', action='store_true', help='khi thoát, không đưa tay máy/thanh trượt về chỗ cũ')
args = ap.parse_args()

rospy.init_node('joint_teleop')
abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
joint_pubs = [rospy.Publisher('/jetmax/joint%d/command' % i, SetJoint, queue_size=1) for i in (1, 2, 3)]
pwm_pub = rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1)
sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)

stepper = None
try:
    import hiwonder
    stepper = hiwonder.Stepper(1)
except Exception as e:
    print('Không khởi tạo được thanh trượt (%s): các phím z/x/c bị tắt.' % e)
slider_enabled = False                   # đã gửi chế độ SLEEP (giữ động cơ) chưa
slider_pos = 0                           # không đọc lại được từ phần cứng nên tự theo dõi, coi lúc mở là 0
rospy.sleep(1.0)


def status():
    return rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)


def joints_of(m):
    return [m.joint1, m.joint2, m.joint3]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def in_limits(x, y, z):
    return (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN)


def show():
    m = status()
    print('  xyz = (%.1f, %.1f, %.1f) | joint = (%.1f, %.1f, %.1f) | cổ tay = %.0f | hút = %s | trượt = %d' % (
        m.x, m.y, m.z, m.joint1, m.joint2, m.joint3, m.pwm1, m.sucker, slider_pos))


def move_xyz(dx, dy, dz):
    m = status()
    cur = (m.x, m.y, m.z)
    t = (clamp(m.x + dx, *X_LIM), clamp(m.y + dy, *Y_LIM), clamp(m.z + dz, *Z_LIM))
    if (t[0] ** 2 + t[1] ** 2) ** 0.5 < R_MIN:
        print('  Quá gần trục (bán kính < %d mm), không đi' % R_MIN)
        return
    dist = sum((t[i] - cur[i]) ** 2 for i in range(3)) ** 0.5
    if dist < 0.1:
        print('  Đã ở biên giới hạn an toàn')
        return
    if abs(dx) + abs(dy) + abs(dz) - dist > 0.5:
        print('  (cắt bớt cho vừa giới hạn an toàn)')
    dur = max(MIN_DUR, dist / SPEED_MM)
    abs_pub.publish(SetJetMax(x=t[0], y=t[1], z=t[2], duration=dur))
    rospy.sleep(dur + 0.1)
    show()


def move_joint(i, delta):
    m = status()
    cur = joints_of(m)[i]
    new = cur + delta
    dur = max(MIN_DUR, abs(delta) / SPEED_DEG)
    joint_pubs[i].publish(SetJoint(data=new, duration=dur))
    rospy.sleep(dur + 0.1)
    m2 = status()
    if abs(joints_of(m2)[i] - new) > JOINT_TOL:
        print('  Lệnh bị từ chối (ngoài giới hạn servo/động học), khớp %d giữ nguyên' % (i + 1))
    elif not in_limits(m2.x, m2.y, m2.z):
        print('  Ra ngoài vùng an toàn, hoàn lại khớp %d' % (i + 1))
        joint_pubs[i].publish(SetJoint(data=cur, duration=dur))
        rospy.sleep(dur + 0.1)
    show()


def move_wrist(delta):
    cur = status().pwm1
    new = int(clamp(cur + delta, PWM_MIN, PWM_MAX))
    pwm_pub.publish(SetServo(data=new, duration=0.2))
    rospy.sleep(0.3)
    show()


def toggle_sucker():
    sucker_pub.publish(Bool(data=not status().sucker))
    rospy.sleep(0.3)
    show()


def arm_high_enough():
    if status().z < SLIDER_MIN_Z:
        print('  Nâng tay máy lên z >= %d trước khi trượt (bấm r)' % SLIDER_MIN_Z)
        return False
    return True


def move_slider(d):
    global slider_pos, slider_enabled
    if stepper is None:
        print('  Thanh trượt không khả dụng')
        return
    if not arm_high_enough():
        return
    new = clamp(slider_pos + d, SLIDER_MIN, SLIDER_MAX)
    d = new - slider_pos
    if d == 0:
        print('  Thanh trượt ở biên (%d)' % slider_pos)
        return
    if not slider_enabled:
        stepper.set_mode(stepper.SLEEP)       # chỉ gửi một lần, không gửi trước mỗi lần bấm
        slider_enabled = True
    stepper.goto(d)
    time.sleep(abs(d) / SLIDER_SPEED + 0.2)
    slider_pos = new
    show()


def slider_home():
    global slider_pos, slider_enabled
    if stepper is None or not arm_high_enough():
        return
    print('  Thanh trượt về home (chờ đến khi xong)...')
    stepper.go_home(True)
    stepper.set_mode(stepper.EN)
    stepper.set_div(stepper.DIV_1_4)
    slider_pos = 0
    slider_enabled = False
    show()


HELP = """
Tịnh tiến (mm):  d/a = x +/-    w/s = y +/-    r/f = z +/-
Khớp (độ):       u/j = joint1   i/k = joint2   o/l = joint3   (n/m = cổ tay)
Thanh trượt:     x/z = tiến/lùi    c = về home thật (đặt lại vị trí 0)
Đi xa hơn:       gõ chữ HOA (giữ Shift) = bước x4, hoặc t để đổi mức bước
v = bật/tắt đầu hút    h = tay máy về home    p = in trạng thái    q = thoát
"""

start = status()
print('Vị trí ban đầu:', (start.x, start.y, start.z), 'pwm1 =', start.pwm1)
print(HELP)
print('Thanh trượt coi như đang ở 0 lúc mở; nếu không chắc, bấm c để về home thật.')
level = 0
print('Bước = %d mm / %d độ / %d bước trượt' % (MM_STEPS[level], DEG_STEPS[level], SLIDER_STEPS[level]))
show()

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)
    while not rospy.is_shutdown():
        if not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        raw = sys.stdin.read(1)
        f = BIG if raw.isupper() else 1
        k = raw.lower()
        mm, deg, sl = MM_STEPS[level] * f, DEG_STEPS[level] * f, SLIDER_STEPS[level] * f
        if k == 'd':
            move_xyz(mm, 0, 0)
        elif k == 'a':
            move_xyz(-mm, 0, 0)
        elif k == 'w':
            move_xyz(0, mm, 0)
        elif k == 's':
            move_xyz(0, -mm, 0)
        elif k == 'r':
            move_xyz(0, 0, mm)
        elif k == 'f':
            move_xyz(0, 0, -mm)
        elif k == 'u':
            move_joint(0, deg)
        elif k == 'j':
            move_joint(0, -deg)
        elif k == 'i':
            move_joint(1, deg)
        elif k == 'k':
            move_joint(1, -deg)
        elif k == 'o':
            move_joint(2, deg)
        elif k == 'l':
            move_joint(2, -deg)
        elif k == 'n':
            move_wrist(deg)
        elif k == 'm':
            move_wrist(-deg)
        elif k == 'x':
            move_slider(sl)
        elif k == 'z':
            move_slider(-sl)
        elif k == 'c':
            slider_home()
        elif k == 'v':
            toggle_sucker()
        elif k == 't':
            level = (level + 1) % len(MM_STEPS)
            print('Bước = %d mm / %d độ / %d bước trượt' % (MM_STEPS[level], DEG_STEPS[level], SLIDER_STEPS[level]))
        elif k == 'h':
            rospy.ServiceProxy('/jetmax/go_home', Empty)()
            rospy.sleep(2.5)
            show()
        elif k == 'p':
            show()
        elif k == 'q':
            break
        termios.tcflush(fd, termios.TCIFLUSH)   # bỏ phím bấm dồn trong lúc robot đang chạy
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    sucker_pub.publish(Bool(data=False))
    if not args.stay:
        abs_pub.publish(SetJetMax(x=start.x, y=start.y, z=start.z, duration=2.0))
        pwm_pub.publish(SetServo(data=int(start.pwm1), duration=1.0))
        rospy.sleep(2.5)
        if stepper is not None and slider_pos != 0:
            if not slider_enabled:
                stepper.set_mode(stepper.SLEEP)
            stepper.goto(-slider_pos)
            time.sleep(slider_pos / SLIDER_SPEED + 0.3)
    if stepper is not None:
        stepper.set_mode(stepper.EN)
    print('Đã thoát' + ('' if args.stay else ' (về vị trí ban đầu)'))
```

---

## Nhóm 3: Thông tin Model & Training

### `training\README.md`

*Dung lượng: 0 B | Số dòng: 0*

```markdown

```

---

