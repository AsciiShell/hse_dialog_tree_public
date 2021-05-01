import multiprocessing
import platform
import re
import subprocess


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    if platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip().decode()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


def get_processor_info():
    cpu_count = multiprocessing.cpu_count()
    cpu_name = get_processor_name()
    return "{name} x {count}".format(count=cpu_count, name=cpu_name)


__all__ = ["get_processor_info"]

if __name__ == '__main__':
    print(get_processor_info())
