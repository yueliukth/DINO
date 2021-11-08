import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class DINOLoss(nn.Module):
    """The loss function.
    We subclass the `nn.Module` becuase we want to create a buffer for the
    logits center of the teacher.
    Parameters
    ----------
    out_dim : int
        The dimensionality of the final layer (we computed the softmax over).
    teacher_temp, student_temp : float
        Softmax temperature of the teacher resp. student.
    center_momentum : float
        Hyperparameter for the exponential moving average that determines
        the center logits. The higher the more the running average matters.
    """
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs,
                 num_epochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.num_crops = num_crops
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(num_epochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """Evaluate loss.
        Parameters
        ----------
        student_output, teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` representing
            logits. The length is equal to number of crops.
            Note that student processed all crops and that the two initial crops
            are the global ones.
        epoch: int
            The current epoch
        Returns
        -------
        loss : torch.Tensor
            Scalar representing the average loss.
        """
        student_output = student_output.chunk(self.ncrops)
        student_temp = [s / self.student_temp for s in student_output]

        # Teacher centering and sharpening
        teacher_output = teacher_output.chunk(2)
        temp = self.teacher_temp_schedule[epoch]
        teacher_temp = [(t - self.center) / temp for t in teacher_output]

        student_softmax = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_softmax = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        n_loss_terms = 0
        for t_ix, t in enumerate(teacher_softmax):
            for s_ix, s in enumerate(student_softmax):
                if t_ix == s_ix:
                    # We skip cases where student and teacher operate on the same view
                    continue

                loss = torch.sum(-t * s, dim=-1)  # (n_samples,)
                total_loss += loss.mean()  # scalar
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.
        Compute the exponential moving average.
        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # To obtain the sum of all tensors at all processes
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
