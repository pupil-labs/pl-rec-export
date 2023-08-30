from pathlib import Path
from typing import Generator, Union

import numpy as np

from . import imu_pb2


def extract_raw_imu_packets(raw_bytes: bytes):
    """
    The neon imu file is a sequence of:

    PACKET_SIZE<uint16>,PACKET_BYTES<packet_size>
    """
    index = 0
    size = len(raw_bytes)
    while index < size:
        nums = np.frombuffer(raw_bytes[index : index + 2], np.uint16)
        if not nums:
            break

        index += 2
        packet_size = nums[0]
        packet_bytes = raw_bytes[index : index + packet_size]
        index += packet_size
        yield packet_bytes


imu_dtype = np.dtype(
    [
        ("unix_time_ns", np.uint64),
        ("gyro_x", np.float32),
        ("gyro_y", np.float32),
        ("gyro_z", np.float32),
        ("accel_x", np.float32),
        ("accel_y", np.float32),
        ("accel_z", np.float32),
        ("quaternion_w", np.float32),
        ("quaternion_x", np.float32),
        ("quaternion_y", np.float32),
        ("quaternion_z", np.float32),
    ]
)


def imu_packets_to_numpy(packets: Generator[imu_pb2.ImuPacket, None, None]):
    data = []
    for packet in packets:
        data.append(
            (
                packet.tsNs,
                packet.gyroData.x,
                packet.gyroData.y,
                packet.gyroData.z,
                packet.accelData.x,
                packet.accelData.y,
                packet.accelData.z,
                packet.rotVecData.w,
                packet.rotVecData.x,
                packet.rotVecData.y,
                packet.rotVecData.z,
            )
        )

    return np.array(data, dtype=imu_dtype)


def raw_imu_file_to_packets(file_path: Union[str, Path]):
    with Path(file_path).open("rb") as handle:
        packets = (
            imu_pb2.ImuPacket.FromString(packet_bytes)
            for packet_bytes in extract_raw_imu_packets(handle.read())
        )
        yield from packets


def raw_imu_file_to_numpy(file_path: Union[str, Path]):
    packet_iterator = raw_imu_file_to_packets(file_path)
    return imu_packets_to_numpy(packet_iterator)
