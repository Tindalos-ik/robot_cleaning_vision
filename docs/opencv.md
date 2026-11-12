# OpenCV

参考文档[1.1. OpenCV-Python教程介绍 - OpenCV Python Tutorials](https://opencv-python-tutorials.readthedocs.io/zh/latest/1. OpenCV简介/1.1. OpenCV-Python教程介绍/)

本项目选用 `opencv`实现连续或按需处理摄像头新采集画面

> OpenCV-Python是一个基于 OpenCV C++ 实现并且封装的 Python 库，可以用于解决计算机视觉问题

针对需求，只需要会图像相关处理，如裁剪，视频入门，会连接摄像头，获取实时画面，以及 OpenCV 的绘图功能即可

优先看：

 - 2.1 图像入门：读取、显示、保存图片。
 - 2.2 视频入门：连接摄像头、获取实时画面。
 - 2.3 OpenCV 中的绘图功能：画检测框、显示文字。
 - 3.1 图像的基本操作：裁剪、像素与图像尺寸。
 - 4.1 改变颜色空间：灰度、BGR、HSV 等。



### 图像入门
 
* 学会如何读取，显示以及保存图像
* api：`cv.imread()`，`cv.imshow()`，`cv.imwrite()`
* 用`matplotlib`显示图像



#### 读取图像

`cv.imread()`用于读取图像，返回一个图像对象，可以用变量接收
* 第一个参数是文件路径参数，可以是相对路径，也可以是绝对路径
* 第二个参数 
  * cv.IMREAD_COLOR  默认参数，以彩色模式加载图像，图像的透明度将被忽略
  * cv.IMREAD_GRAYCALE 以灰度模式加载图像
  * cv.IMREAD_UNCHANGED 以 alpha 通道模式加载图像
也可以传递 1，0，-1来代替上面三个参数

```python
i
`mport numpy as np``
import cv2 as cv

img = cv.imread('messi5.jpg', 0)


即使图像路径错误，它也不会报错，但是`print(img)`会显示`None`


#### 显示图像

`cv.imshow()`用于在窗口中显示图像，窗口会自动适应图像大小
 * 第一个参数是窗口名称，字符串
 * 第二个参数是要加载的图像，你可以显示多个图像窗口，但是每个窗口名称必须不同

```python
# 显示图像
cv.imshow('image', img)
cv.waitKey(0) # 等待键盘输入，不然直接结束了
cv.destroyAllWindows()
```

`cv.waitkey()`是一个键盘事件函数，它的参数以毫秒为单位，该函数在毫秒的时间内去等待键盘事件，如果有键盘事件触发则程序继续，参数设置为0，则一直等待。它也可以设置为检测指定按键的触发

`cv.destoryAllWindow()`关于所有窗口，也可以使用`cv.destoryWindow(img)`关闭指定窗口

我们也可以先创建一个窗口，再加载图像

```python
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()

```



#### 保存图像
 
`cv.imwrite()`函数用于保存图像
* 第一个参数，保存为的图片名
* 第二个参数，待保存的图像

```python
cv.imwrite('pikachu.png',img)
```


#### 总结

下面的代码程序将图片加载并显示为灰度图像，按`s`保存图像并退出，按`esc`直接退出

```python
import numpy as np
import cv2 as cv
img = cv.imread('messi5.jpg',0)
cv.imshow('image',img)
k = cv.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv.imwrite('messigray.png',img)
    cv.destroyAllWindows()
```


### 视频入门
 
* 学习读取，显示和保存视频
* 用摄像头捕获并显示视频
* api：`cv.VideoCapture()`, `cv.VideoWriter()`


#### 用摄像头捕获视频

通常，我们需要使用摄像头来捕获直播画面，OpenCV 为此提供了非常简单的函数接口

下面我们尝试使用摄像头来捕获视频画面，并将画面转换成灰度图像显示

首先创建一个`VideoCapture`对象，它的参数可以是设备索引或者是视频文件的名称
* 设备索引就是设备所对应的设备号，当只连接一个摄像头时，参数只需传递0(或-1)
可以通过一些方法连接到别的摄像头，比如手机摄像头

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    # 逐帧读取视频
    ret, frame = cap.read()

    # 转换为灰度图像
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 显示图像
    cv.imshow('frame', gray)

    # 按下q键退出循环
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv.destroyAllWindows()
```

`cap.read()`返回一个 bool 值

如果初始化摄像头失败，上面的代码会报错，可以使用`cap.isOpened()`来检查是否初始化，如果初始化失败，使用`cap.open()`

 ```python
# 将创建对象和打开摄像头分开
cap = cv.VideoCapture()
if not cap.open(0):
    print("无法打开摄像头")
 ```


#### 播放视频文件

与从相机捕获视频原理相同，只需将设备索引更改成视频文件的名字即可。同时在x
显示帧时，给`cv.waitKey()`传递适当的时间参数，如果它太小，视频会很快，太高又很慢，正常 25ms 即可

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```

播放需要`ffmpeg`或`gstreamer`


#### 保存视频

首先创建一个`VideoWriter`对象，指定输出文件名，然后指定 FourCC 代码并传递每秒帧数和帧大小，最后是`isColor`标志，True表示彩色

FourCC 是用于指定视频编解码器的4字节代码

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
```


### 绘图功能

* 用 OpenCV 绘制图形
* api：`cv.line()`, `cv.circle()`, `cv.rectangle()`, `cv.ellipse()`, `cv.putText()`
参数
  * img，设置图像
  * color 颜色，一个RGB值，用元组传递
  * thickness 用于设置线条或圆形等的厚度
  * lineType 设置线条类型


#### 绘制直线

绘制线条，需要设置起点和终点坐标，我们将创建一个黑色图像，并在其上从左上角到右下角绘制一条蓝线

```python
import numpy as np
import cv2 as cv

img = np.zeros((512,512,3), np.uint8)
cv.line(img, (0,0), (511,511), (255,0,0), 5)

cv.imshow('image', img)
cv.waitKey(0)
```


#### 绘制矩形
 
设置左上角和右下角坐标即可

```python
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
```


#### 绘制圆形

设置圆点坐标和半径

```python
cv.circle(img,(447,63), 63, (0,0,255), -1)
```

向图像添加文字

### 
```python
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
```


## 核心特性



* 获取并修改图像的像素值
* 获取图像的属性



## 一些概念


### BGR

BGR 图像是指颜色通道顺序为 Blue，Green，Red 的图像

普通图像是 RGB 顺序，而 OpenCV 的`cv.imread()`读进来是 BGR 顺序





