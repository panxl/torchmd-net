from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict


class QM9(QM9_geometric):
    def __init__(self, root, label, transform=None, *args, **kwargs):
        self.label = label
        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM9, self).__init__(root, transform=transform, *args, **kwargs)

    def get_atomref(self):
        return self.atomref(self.label_idx)
    
    def _filter_label(self, batch):
        batch.y = batch.y[:,self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        super(QM9, self).download()

    def process(self):
        super(QM9, self).process()