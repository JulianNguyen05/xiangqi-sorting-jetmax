# GATHERED SOURCE CODE

- **Thư mục gốc:** `D:\2026_2027\Chuyen-de\xiangqi-sorting-jetmax`
- **Tổng số file gôm được:** 8/11

### ⚠️ Các file chưa tìm thấy:
- `jetmax.py`
- `stepper.py`
- `pwm_servo.py`

## Mục lục

- [docs\robot-reference\waste_classification_stepper.py](#docsrobot-referencewaste_classification_stepperpy)
- [docs\robot-reference\jetmax_control\scripts\jetmax_control_main.py](#docsrobot-referencejetmax_controlscriptsjetmax_control_mainpy)
- [docs\robot-reference\jetmax_control\msg\SetJetMax.msg](#docsrobot-referencejetmax_controlmsgsetjetmaxmsg)
- [docs\robot-reference\jetmax_control\msg\JetMax.msg](#docsrobot-referencejetmax_controlmsgjetmaxmsg)
- [docs\robot-reference\jetmax_control\msg\SetServo.msg](#docsrobot-referencejetmax_controlmsgsetservomsg)
- [tests\test_move.py](#teststest_movepy)
- [tests\capture_images.py](#testscapture_imagespy)
- [tests\stepper_teleop.py](#testsstepper_teleoppy)

---

### `docs\robot-reference\waste_classification_stepper.py`

*Số dòng: 283*

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

### `docs\robot-reference\jetmax_control\scripts\jetmax_control_main.py`

*Số dòng: 182*

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

### `docs\robot-reference\jetmax_control\msg\SetJetMax.msg`

*Số dòng: 4*

```text
float32 x
float32 y
float32 z
float32 duration
```

---

### `docs\robot-reference\jetmax_control\msg\JetMax.msg`

*Số dòng: 12*

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

### `docs\robot-reference\jetmax_control\msg\SetServo.msg`

*Số dòng: 2*

```text
uint16 data
float32 duration
```

---

### `tests\test_move.py`

*Số dòng: 48*

```python
#!/usr/bin/env python3
# Chỉ đi ±20 mm theo x, y, z từ vị trí ban đầu (z đi xuống 20 mm rồi lên lại).
# Khi kết thúc hoặc bấm q, robot tự về chỗ cũ và tắt đầu hút.

import rospy
from std_msgs.msg import Bool
from jetmax_control.msg import SetJetMax, JetMax

STEP = 20.0   # mm mỗi bước
DUR = 2.0     # giây mỗi bước

rospy.init_node('test_move')
abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
rel_pub = rospy.Publisher('/jetmax/relative_command', SetJetMax, queue_size=1)
sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
rospy.sleep(1.0)

def get_pos():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return m.x, m.y, m.z

def ask(text):
    return input(text + ' [Enter = chạy, q = dừng] ').strip().lower() != 'q'

start = get_pos()
print('Vị trí ban đầu x,y,z =', start)

steps = [
    ('x +', (STEP, 0, 0)), ('x - (về)', (-STEP, 0, 0)),
    ('y +', (0, STEP, 0)), ('y - (về)', (0, -STEP, 0)),
    ('z - (xuống)', (0, 0, -STEP)), ('z + (về)', (0, 0, STEP)),
]
try:
    for name, (dx, dy, dz) in steps:
        if not ask('Bước: ' + name):
            break
        rel_pub.publish(SetJetMax(x=dx, y=dy, z=dz, duration=DUR))
        rospy.sleep(DUR + 0.5)
        print('  ->', get_pos())

    if ask('Test đầu hút (bật 2 giây)'):
        sucker_pub.publish(Bool(data=True))
        rospy.sleep(2)
finally:
    sucker_pub.publish(Bool(data=False))
    abs_pub.publish(SetJetMax(x=start[0], y=start[1], z=start[2], duration=2.0))
    rospy.sleep(2.5)
    print('Đã về vị trí ban đầu')
```

---

### `tests\capture_images.py`

*Số dòng: 56*

```python
#!/usr/bin/env python3

# Bấm phím cách để lưu, q để thoát. Ảnh lưu vào ~/capture/,
# mỗi lần lưu ghi thêm một dòng vào positions.csv (tên file, x, y, z).

import os, time, csv
import cv2, numpy as np, rospy
from sensor_msgs.msg import Image
from jetmax_control.msg import JetMax

TOPIC = '/usb_cam/image_rect_color'
OUT = os.path.expanduser('~/capture')
os.makedirs(OUT, exist_ok=True)
state = {'img': None, 'pos': (0.0, 0.0, 0.0)}

def on_image(msg):
    state['img'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()  # RGB

def on_status(msg):
    state['pos'] = (msg.x, msg.y, msg.z)

rospy.init_node('capture_images')
rospy.Subscriber(TOPIC, Image, on_image, queue_size=1)
rospy.Subscriber('/jetmax/status', JetMax, on_status, queue_size=1)

csv_path = os.path.join(OUT, 'positions.csv')
is_new = not os.path.exists(csv_path)
f = open(csv_path, 'a', newline='')
w = csv.writer(f)
if is_new:
    w.writerow(['file', 'x', 'y', 'z'])

n, t0 = 0, time.time()
print('Space = lưu ảnh, q = thoát')
while not rospy.is_shutdown():
    if state['img'] is None:
        if time.time() - t0 > 5:
            print('Chưa nhận được ảnh từ', TOPIC, '- kiểm tra rostopic list')
            t0 = time.time()
        time.sleep(0.05)
        continue
    bgr = cv2.cvtColor(state['img'], cv2.COLOR_RGB2BGR)
    show = bgr.copy()
    cv2.putText(show, 'saved: %d' % n, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('capture', show)
    k = cv2.waitKey(30) & 0xFF
    if k == ord(' '):
        name = '%d.jpg' % int(time.time() * 1000)
        cv2.imwrite(os.path.join(OUT, name), bgr)
        w.writerow([name] + ['%.1f' % v for v in state['pos']])
        f.flush()
        n += 1
    elif k == ord('q'):
        break
f.close()
cv2.destroyAllWindows()
```

---

### `tests\stepper_teleop.py`

*Số dòng: 83*

```python
#!/usr/bin/env python3
# Dieu khien thanh truot (stepper) cua JetMax bang ban phim trong terminal.
# Dua theo cach dung hiwonder.Stepper trong waste_classification_stepper.py cua hang:
#   - goto(n) di chuyen TUONG DOI n buoc (goto(-n) de quay lai), thoi gian ~ n/1000 giay
#   - vi tri 0 = diem home, vi du cua hang di toi da khoang 5200 buoc tu home
import sys, select, termios, tty, time
import rospy
import hiwonder

POS_MIN, POS_MAX = 0, 5200      # gioi han phan mem (buoc), 0 = home
STEPS = [200, 500, 1000]        # cac muc buoc
step_i = 0                      # bat dau bang buoc nho nhat

rospy.init_node('stepper_teleop')
stepper = hiwonder.Stepper(1)

print('Se dua thanh truot ve HOME (vi tri 0). Hay dam bao canh tay dang nang cao va khong vuong vat can.')
input('Bam Enter de bat dau, Ctrl+C de huy: ')
stepper.go_home(True)
stepper.set_mode(stepper.EN)
stepper.set_div(stepper.DIV_1_4)
pos = 0
print('Da o HOME, vi tri = 0')


def move(delta):
    global pos
    new = max(POS_MIN, min(POS_MAX, pos + delta))
    d = new - pos
    if d == 0:
        print('Cham gioi han, khong di nua (vi tri = %d)' % pos)
        return
    stepper.set_mode(stepper.SLEEP)
    stepper.goto(d)
    time.sleep(abs(d) / 1000.0 + 0.3)
    pos = new
    print('vi tri = %d / %d' % (pos, POS_MAX))


def go_home():
    global pos
    stepper.go_home(True)
    stepper.set_mode(stepper.EN)
    stepper.set_div(stepper.DIV_1_4)
    pos = 0
    print('Da ve HOME, vi tri = 0')


HELP = """
l : di ra xa home (+)      j : ve gan home (-)
1/2/3 : buoc 200/500/1000  h : ve home
p : in vi tri (uoc luong theo phan mem)
q : thoat (nha khoa thanh truot)
"""
print(HELP)
print('Buoc = %d' % STEPS[step_i])

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)
    while not rospy.is_shutdown():
        if not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        k = sys.stdin.read(1).lower()
        s = STEPS[step_i]
        if k == 'l':
            move(s)
        elif k == 'j':
            move(-s)
        elif k in ('1', '2', '3'):
            step_i = int(k) - 1
            print('Buoc = %d' % STEPS[step_i])
        elif k == 'h':
            go_home()
        elif k == 'p':
            print('vi tri = %d (uoc luong)' % pos)
        elif k == 'q':
            break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    stepper.set_mode(stepper.EN)
    print('Da thoat')
```

---

