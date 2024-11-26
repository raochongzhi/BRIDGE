import torch
from torch.utils.data import Dataset


class UserJobMatchingDataset(Dataset):
    def __init__(self, data, max_history_length, feature_dim):
        """
        :param data: List of samples, each sample is a dictionary with:
                     {
                         "target_user": [user features],
                         "history_users": [[user features], ...],
                         "target_job": [job features],
                         "history_jobs": [[job features], ...],
                         "label": 0 or 1
                     }
        :param max_history_length: int, maximum history length to pad or truncate the sequences.
        """
        self.data = data
        self.max_history_length = max_history_length
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        target_user = sample["target_user"].clone().detach().float()
        target_job = sample["target_job"].clone().detach().float()

        history_users = sample["history_users"]
        history_jobs = sample["history_jobs"]
        # print(history_users)
        history_users_padded = self.pad_or_truncate(history_users, self.max_history_length)
        history_jobs_padded = self.pad_or_truncate(history_jobs, self.max_history_length)

        sequence_mask_users = self.generate_mask(len(history_users), self.max_history_length)
        sequence_mask_jobs = self.generate_mask(len(history_jobs), self.max_history_length)

        history_users_padded = torch.stack(history_users_padded, dim=0)
        history_jobs_padded = torch.stack(history_jobs_padded, dim=0)
        sequence_mask_users = torch.tensor(sequence_mask_users, dtype=torch.float32)
        sequence_mask_jobs = torch.tensor(sequence_mask_jobs, dtype=torch.float32)

        label = torch.tensor(sample["label"], dtype=torch.float32)

        inputs = {
            'target_user': target_user,
            'history_users': history_users_padded,
            'sequence_mask_users': sequence_mask_users,
            'target_job': target_job,
            'history_jobs': history_jobs_padded,
            'sequence_mask_jobs': sequence_mask_jobs,
            'labels': label
        }

        return inputs

    def pad_or_truncate(self, sequence, max_length):
        """
        Pads or truncates a sequence to the max_length.
        :param sequence: List of sequences (e.g., user or job features).
        :param max_length: The target length for the sequence.
        :return: Padded or truncated sequence (a 2D list where each sub-list has the same size).
        """
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            # feature_length = len(sequence[0]) if sequence else self.feature_dim
            feature_length = self.feature_dim
            padding = [torch.tensor([0.0] * feature_length)] * (max_length - len(sequence))
            return sequence + padding

    def generate_mask(self, seq_length, max_length):
        """
        Generates a mask for the sequence. 1 for valid positions, 0 for padded positions.
        :param seq_length: Actual sequence length.
        :param max_length: Target sequence length.
        :return: A mask list indicating valid and padded positions.
        """
        return [1] * min(seq_length, max_length) + [0] * (max_length - min(seq_length, max_length))