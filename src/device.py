import os

devices = ['CPU']


class _Device:
    @staticmethod
    def DEFAULT():
        for device in devices:
            if os.environ.get(device) == '1':
                return device
        return 'CPU'


Device = _Device()
