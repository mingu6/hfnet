# Allocate INS poses

For the RobotCar datasets, there are 3 synchronised cameras (left, right, rear) on the vehicle. The `gps_allocation_robotcar.py` program interpolates the raw INS data to camera timestamps and applies camera extrinsics to yield the full 6DoF pose with respect to a common global frame. 

This program saves the object for the full traverse as `DATA_PATH/robotcar/gps/{traverse_name}.pickle`, with the contents being a `SE3Poses` object found in `QUT/geometry.py`. 
