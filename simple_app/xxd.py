# -*- coding: utf-8 -*-
"""Converts tflite models into hex-code
python2.7 xxd.py gait_cnn.tflite > gait_cnn.cc
"""
import os.path
import string
import sys

def xxd(file_path):
    with open(file_path, 'r') as f:
        array_name = file_path.replace('/','_').replace('.','_')
        output = "unsigned char %s[] = {" % array_name
        length = 0
        while True:
            buf = f.read(12)

            if not buf:
                output = output[:-2]
                break
            else:
                output += "\n  "

            for i in buf:
                output += "0x%02x, " % ord(i)
                length += 1
        output += "\n};\n"
        output += "unsigned int %s_len = %d;" % (array_name, length)
        print output


if __name__ == '__main__':
    if not os.path.exists(sys.argv[1]):
        print >> (sys.stderr, "The file doesn't exist.")
        sys.exit(1)
    xxd(sys.argv[1])