
import torch

def random_rotation_matrix():
    # Generate a random quaternion
    q = torch.randn(4)
    q /= torch.norm(q)  # Normalize the quaternion

    # Convert quaternion to rotation matrix
    q0, q1, q2, q3 = q.unbind()
    R = torch.tensor([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0],
        [2*q1*q2 + 2*q3*q0, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q1*q0],
        [2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1**2 - 2*q2**2]
    ])
    return R

class RandomProjection:
    """
    Apply a random projection to the input batch.

    By default only the line-of-sight velocity (the component along the
    axis that gets removed from the position) is kept, mimicking a
    radial-velocity-only observation. Setting ``use_proper_motions=True``
    additionally keeps the two velocity components in the sky plane,
    i.e. the proper motion expressed in km/s, alongside the
    line-of-sight velocity.

    Columns are always ordered cyclically starting from the line-of-sight
    axis: if ``axis`` is the LOS axis, the remaining axes are taken in the
    order ``(axis + 1) % 3, (axis + 2) % 3`` for both ``pos`` (sky-plane
    position) and ``vel`` (line-of-sight velocity first, followed by the
    two proper-motion components, when ``use_proper_motions=True``).
    """
    def __init__(self, axis=None, use_proper_motions=False):
        self.axis = axis
        self.use_proper_motions = use_proper_motions

    def __call__(self, batch):
        batch = batch.clone()

        if self.axis is None:
            # create the random projection matrix
            R = random_rotation_matrix()

            # apply rotation to position and velocity
            pos = torch.matmul(batch.pos, R)
            vel = torch.matmul(batch.vel, R)

            # by convention the observer looks along the last rotated axis
            axis = 2
        else:
            pos = batch.pos
            vel = batch.vel
            axis = self.axis

        # cyclic ordering of the two axes orthogonal to the LOS axis
        other1 = (axis + 1) % 3
        other2 = (axis + 2) % 3

        pos_proj = torch.stack([pos[:, other1], pos[:, other2]], dim=1)

        if self.use_proper_motions:
            # line-of-sight velocity first, followed by the two
            # proper-motion components, in cyclic order
            vel_proj = torch.stack(
                [vel[:, axis], vel[:, other1], vel[:, other2]], dim=1
            )
        else:
            vel_proj = vel[:, axis].unsqueeze(1)

        # update the batch
        batch.pos = pos_proj
        batch.vel = vel_proj

        return batch
