
```sh
sudo ffplay -f v4l2 -framerate 20 -video_size 640x360 -input_format yuyv422 -i /dev/video0

scrcpy
adb devices
ls /dev/video*

bash 2_openapp.sh
```
