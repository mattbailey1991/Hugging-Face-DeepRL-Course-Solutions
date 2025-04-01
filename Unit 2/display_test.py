import os
os.environ['PYVIRTUALDISPLAY_DISPLAYFD'] = '0'

from easyprocess import EasyProcess

from pyvirtualdisplay import Display

with Display(visible=True, size=(100, 60)) as disp:
    with EasyProcess(["xmessage", "hello"]) as proc:
        proc.wait()