from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AccelData(_message.Message):
    __slots__ = ["reserved", "x", "y", "z"]
    RESERVED_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    reserved: int
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., reserved: _Optional[int] = ...) -> None: ...

class GyroData(_message.Message):
    __slots__ = ["reserved", "x", "y", "z"]
    RESERVED_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    reserved: int
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., reserved: _Optional[int] = ...) -> None: ...

class ImuPacket(_message.Message):
    __slots__ = ["accelData", "gyroData", "rotVecData", "tsNs"]
    ACCELDATA_FIELD_NUMBER: _ClassVar[int]
    GYRODATA_FIELD_NUMBER: _ClassVar[int]
    ROTVECDATA_FIELD_NUMBER: _ClassVar[int]
    TSNS_FIELD_NUMBER: _ClassVar[int]
    accelData: AccelData
    gyroData: GyroData
    rotVecData: RotVecData
    tsNs: int
    def __init__(self, tsNs: _Optional[int] = ..., accelData: _Optional[_Union[AccelData, _Mapping]] = ..., gyroData: _Optional[_Union[GyroData, _Mapping]] = ..., rotVecData: _Optional[_Union[RotVecData, _Mapping]] = ...) -> None: ...

class RotVecData(_message.Message):
    __slots__ = ["reserved", "w", "x", "y", "z"]
    RESERVED_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    reserved: float
    w: float
    x: float
    y: float
    z: float
    def __init__(self, w: _Optional[float] = ..., x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., reserved: _Optional[float] = ...) -> None: ...
