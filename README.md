
```sh
# cmd for using ffplay to check the UVC stream
sudo ffplay -f v4l2 -framerate 20 -video_size 640x360 -input_format yuyv422 -i /dev/video0

# basic cmd
scrcpy
adb devices
ls /dev/video*
bash 2_openapp.sh

# file detail
1_runuvctool.sh => 2_openapp.sh + pyCVTest5p.py + 3_CloseApp.sh
```

## ros utils
### topics
```bash
/Altek/depth/image_rect_raw

/Altek/color/image_raw
/AltekC/color/image_raw/compressed
```

## basic operation
```python
ret, frame_cv = cap.read()
if ret == 0 :
    print("No video frame")
    ch = cv2.waitKey(g_wait_1s)
    continue
```

## Hardware setting
![](https://github.com/wowyunDBL/pyCVTest_ub/blob/master/Image/setting.jpg)