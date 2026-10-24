from pathlib import Path
import tempfile
import unittest

from PIL import Image
from dataset.CT import CT


class CTSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ('COVID_tr', 'COVID_ts', 'nonCOVID_tr', 'nonCOVID_ts'):
            target = self.root / 'CT' / folder
            target.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(target / 'sample.png')

    def test_training_and_evaluation_do_not_share_paths(self):
        train, evaluation = CT(str(self.root), 'train'), CT(str(self.root), 'eval')
        self.assertFalse(set(train.im_paths) & set(evaluation.im_paths))
        self.assertEqual({Path(p).parent.name for p in train.im_paths}, {'COVID_tr', 'nonCOVID_tr'})
        self.assertEqual({Path(p).parent.name for p in evaluation.im_paths}, {'COVID_ts', 'nonCOVID_ts'})

    def test_biological_classes_keep_the_same_labels_across_splits(self):
        for mode in ('train', 'eval'):
            data = CT(str(self.root), mode)
            labels = {Path(p).parent.name: y for p, y in zip(data.im_paths, data.ys)}
            suffix = 'tr' if mode == 'train' else 'ts'
            self.assertEqual(labels[f'COVID_{suffix}'], 0)
            self.assertEqual(labels[f'nonCOVID_{suffix}'], 1)
            self.assertEqual(data.nb_classes(), 2)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            CT(str(self.root), 'validation')


if __name__ == '__main__':
    unittest.main()
