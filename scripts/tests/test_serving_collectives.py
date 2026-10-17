import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('collectives', Path(__file__).parents[1] / 'summarize-serving-collectives.py')
collectives = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collectives)


class CollectivesTest(unittest.TestCase):
    def test_payload_is_per_call_and_unknown_dtype_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / 'hccl_single_device.db'
            with sqlite3.connect(table) as connection:
                connection.execute('create table HCCLOpSingleDevice(op_type,alg_type,data_type,count,rank_size)')
                connection.executemany('insert into HCCLOpSingleDevice values(?,?,?,?,?)',
                    [('reduce','ring','FP32',10,8)]*2 + [('gather','ring','unknown',11,8)])
            report = collectives.summarize(table)
            gather, reduce = report['collectives']
            self.assertIsNone(gather['input_bytes_per_call'])
            self.assertEqual(reduce['calls'], 2)
            self.assertEqual(reduce['elements_per_call'], 10)
            self.assertEqual(reduce['input_bytes_per_call'], 40)
            with sqlite3.connect(table) as connection:
                connection.execute('insert into HCCLOpSingleDevice values(?,?,?,?,?)', ('bad','ring','FP32',-1,8))
            with self.assertRaises(ValueError):
                collectives.summarize(table)


if __name__ == '__main__':
    unittest.main()
