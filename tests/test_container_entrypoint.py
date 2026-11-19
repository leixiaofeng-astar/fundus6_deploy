"""Configuration behavior that must fail before model startup in containers."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('container_entrypoint', Path(__file__).resolve().parents[1]/'docker/entrypoint.py')
ENTRY=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


class ContainerConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        for name in ['weights/mermed6_ep41.pth','fundus_IQA/eyeqc_class_model4vit_base_lr1e-06-AUC9988-3DB-statedict.pth']:
            file=self.root/name;file.parent.mkdir(exist_ok=True);file.write_bytes(b'test')
        self.env={'FUNDUS_API_KEY':'a-private-example-secret-for-tests', 'FUNDUS_MODEL_ROOT':str(self.root),
                  'FUNDUS_DATA_DIR':str(self.root/'data')}

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_cpu_default(self):
        self.assertEqual(ENTRY.validate(self.env),'cpu')
        self.assertTrue((self.root/'data').is_dir())

    def test_missing_or_placeholder_secret_fails_closed(self):
        for value in ['', 'short', 'replace-with-a-long-random-secret']:
            self.env['FUNDUS_API_KEY']=value
            with self.assertRaises(ValueError):ENTRY.validate(self.env)

    def test_missing_checkpoint_reports_mount(self):
        (self.root/'weights/mermed6_ep41.pth').unlink()
        with self.assertRaisesRegex(ValueError,'Mount the weights'):
            ENTRY.validate(self.env)

    def test_gpu_selection_and_invalid_device(self):
        self.env['FUNDUS_DEVICE']='cuda'
        self.assertEqual(ENTRY.validate(self.env),'cuda')
        self.env['FUNDUS_DEVICE']='anything'
        with self.assertRaises(ValueError):ENTRY.validate(self.env)

    def test_separate_bearer_secret_is_validated(self):
        self.env['FUNDUS_BEARER_TOKEN']='short'
        with self.assertRaises(ValueError):ENTRY.validate(self.env)


if __name__=='__main__':unittest.main()
