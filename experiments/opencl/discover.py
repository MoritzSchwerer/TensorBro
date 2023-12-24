import pyopencl as cl

platforms = cl.get_platforms()

for p in platforms:
    print(f"Platform: {p.name}")
    for device in p.get_devices():
        print(f"    Device: {device.name}")
