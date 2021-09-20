
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
## Hardware setting
![](https://github.com/wowyunDBL/pyCVTest_ub/blob/master/Image/setting.jpg)