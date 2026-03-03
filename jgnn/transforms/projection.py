
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
    """ Apply a random projection to the input batch """
    def __init__(self, axis=None):
        self.axis = axis

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
            vel_proj = vel_proj[:, 2].unsqueeze(1)
        else:
            pos_proj = torch.cat([batch.pos[:, :self.axis], batch.pos[:, self.axis+1:]], dim=1)
            vel_proj = batch.vel[:, self.axis].unsqueeze(1)

        # update the batch
        batch.pos = pos_proj
        batch.vel = vel_proj

        return batch
