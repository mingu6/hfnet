import numpy as np
import math

from scipy.spatial.transform import Rotation

class SE3Poses:
    def __init__(self, t, R):
        self._single = False

        if t.ndim not in [1, 2] or t.shape[-1] != 3:
            raise ValueError("Expected `t` to have shape (3,) or (N x 3), "
                             "got {}.".format(t.shape))

        # If a single quaternion is given, convert it to a 2D 1 x 4 matrix but
        # set self._single to True so that we can return appropriate objects
        # in the `to_...` methods

        if t.shape == (3,):
            t = t[None, :]
            self._single = True
            if len(R) > 1:
                raise ValueError("Different number of translations 1 and 
                                 rotations {}.".format(len(R)))
        elif len(t) == 1:
            self._single = True
        else:
            if len(t) != len(R):
                raise ValueError("Differing number of translations {} and 
                                 rotations {}".format(len(t),len(R)))
        self._t = t
        self._R = R
        self.len = len(R)

    def __getitem__(self, indexer):
        return self.__class__(self.t()[indexer], self.R()[indexer])

    def __len__(self):
        return self.len

    def __mul__(self, other):
        """
        Performs element-wise pose composition.
        TO DO: Broadcasting
        """
        if not(len(self) == 1 or len(other) == 1 or len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in both "
                             "or a single transformation in either object, "
                             "got {} transformations in first and {} 
                             transformations in second object.".format(
                                len(self), len(other)))
        return self.__class__(self.R().apply(other.t()) + self.t(), i
                              self.R() * other.R())

    def __truediv__(self, other):
        """
        Computes relative pose, similar to MATLAB convention 
        (x = A \ b for Ax = b). Example: T1 / T2 = T1.inv() * T2
        TO DO: Broadcasting
        """
        if not(len(self) == 1 or len(other) == 1 or len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in both "
                             "or a single transformation in either object, "
                             "got {} transformations in first and {} 
                             transformations in second object.".format(
                                len(self), len(other)))
        R1_inv = self.R().inv()
        t_new = R1_inv.apply(other.t() - self.t())
        return self.__class__(t_new, R1_inv * other.R())

    def t(self):
        return self._t[0] if self._single else self._t

    def R(self):
        return self._R

    def inv(self):
        R_inv = self.R().inv()
        t_new = -R_inv.apply(self.t())
        return SE3Poses(t_new, R_inv)

    def components(self):
        """
        Return translational and rotational components of pose separately. 
        Quaternion form for rotations.
        """
        return self.t(), self.R()

    def repeat(self, N):
        t = self.t()
        q = self.R().as_quat()
        if len(self) == 1:
            t = np.expand_dims(t, 0)
            q = np.expand_dims(q, 0)
        t = np.repeat(t, N, axis=0)
        q = np.repeat(q, N, axis=0)
        return SE3Poses(t, Rotation.from_quat(q))

def metric(p1, p2, w):
    """
    Computes metric on the cartesian product space representation of SE(3).
    Args:
        p1 (SE3Poses) : set of poses
        p2 (SE3Poses) : set of poses (same size as p1)
        w (float > 0) : weight for attitude component
    """
    if not(len(p1) == 1 or len(p2) == 1 or len(p1) == len(p2)):
        raise ValueError("Expected equal number of transformations in both "
                            "or a single transformation in either object, "
                            "got {} transformations in first and {} 
                            transformations in second object.".format(
                            len(p1), len(p2)))
    if w < 0:
        raise ValueError("Weight must be non-negative, currently {}".format(w))
    p_rel = p1 / p2
    t_dist = np.linalg.norm(p_rel.t(), axis=-1)
    R_dist = p_rel.R().magnitude()
    return t_dist + w * R_dist 

def combine(listOfPoses):
    """
    combines a list of SE3 objects into a single SE3 object
    """
    tList = []
    qList = []
    for pose in listOfPoses:
        if len(pose) == 1:
            t_temp = np.expand_dims(pose.t(), 0)
            q_temp = np.expand_dims(pose.R().as_quat(), 0)
        else:
            t_temp = pose.t()
            q_temp = pose.R().as_quat()
        tList.append(t_temp)
        qList.append(q_temp)
    tList = np.concatenate(tList, axis=0)
    qList = np.concatenate(qList, axis=0)
    return SE3Poses(tList, Rotation.from_quat(qList))
