
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
            pos_proj = torch.matmul(batch.pos, R)
            vel_proj = torch.matmul(batch.vel, R)

            # apply the projection by removing the last dimension
            pos_proj = pos_proj[:, :2]
            if not self.use_proper_motions:
                # keep only the line-of-sight velocity component
                vel_proj = vel_proj[:, 2].unsqueeze(1)
            # otherwise keep all 3 rotated velocity components: the first
            # two match pos_proj (proper motion in km/s), the last one is
            # the line-of-sight velocity
        else:
            pos_proj = torch.cat([batch.pos[:, :self.axis], batch.pos[:, self.axis+1:]], dim=1)
            if self.use_proper_motions:
                # keep all 3 velocity components: proper motion (the two
                # axes orthogonal to `axis`) plus the line-of-sight
                # velocity along `axis`
                vel_proj = batch.vel
            else:
                vel_proj = batch.vel[:, self.axis].unsqueeze(1)

        # update the batch
        batch.pos = pos_proj
        batch.vel = vel_proj

        return batch
