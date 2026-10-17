import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cases', Path(__file__).parents[1] / 'make-long-context-cases.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LongContextCases(unittest.TestCase):
    def test_exact_envelope_and_needle(self):
        contexts = [8192, 16384, 32768, 65536]
        positions = ['early', 'middle', 'late']
        cases = module.make_cases(lambda text: list(text.encode()), contexts, positions)
        self.assertEqual(len(cases), 12)
        self.assertEqual(cases, module.make_cases(lambda text: list(text.encode()), contexts, positions))
        answers = set()
        for case in cases:
            q = case['qualification']
            self.assertEqual(len(case['input_ids']) + len(case['output_ids']), q['total_context'])
            start, end = q['needle_token_span']
            needle = bytes(case['input_ids'][start:end]).decode()
            self.assertIn(q['expected_answer'], needle)
            self.assertEqual(bytes(case['input_ids']).count(q['expected_answer'].encode()), 1)
            self.assertTrue(q['output_is_placeholder'])
            answers.add(q['expected_answer'])
        self.assertEqual(len(answers), 12)
        for offset in range(0, 12, 3):
            starts = [c['qualification']['needle_token_span'][0] for c in cases[offset:offset+3]]
            self.assertEqual(starts, sorted(set(starts)))

    def test_reject_invalid_envelopes(self):
        for contexts, positions, output in [([65537], ['early'], 16), ([1], ['late'], 16),
                ([8192], ['early'], 0), ([], ['early'], 16), ([8192], [], 16),
                ([8192, 8192], ['early'], 16), ([8192], ['late', 'late'], 16),
                ([8192], ['elsewhere'], 16)]:
            with self.subTest(contexts=contexts, positions=positions, output=output):
                with self.assertRaises(ValueError):
                    module.make_cases(lambda text: list(text.encode()), contexts, positions, output)

    def test_reject_empty_tokenizer_template(self):
        with self.assertRaises(ValueError):
            module.make_cases(lambda text: [], [8192], ['early'])

    def test_comparison_rejects_placeholders(self):
        compare_spec = importlib.util.spec_from_file_location('compare', Path(__file__).parents[1] / 'compare-program-reference.py')
        compare = importlib.util.module_from_spec(compare_spec)
        compare_spec.loader.exec_module(compare)
        cases = module.make_cases(lambda text: list(text.encode()), [8192], ['early'])
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            compare.compare(cases, [], 1, 148902)

    def test_equal_wrong_answers_do_not_pass_retrieval(self):
        compare_spec = importlib.util.spec_from_file_location('compare', Path(__file__).parents[1] / 'compare-program-reference.py')
        compare = importlib.util.module_from_spec(compare_spec)
        compare_spec.loader.exec_module(compare)
        cases = [dict(qualification=dict(expected_answer='1234'))]
        result = [dict(reference_output_ids=[9, 0, 8], native_output_ids_by_repetition=[[9, 0, 8]])]
        compare.check_answers(cases, result, lambda ids: 'wrong' if ids == [9] else '1234', 0)
        self.assertFalse(result[0]['retrieval']['reference_correct'])
        self.assertFalse(result[0]['retrieval']['native_correct'])
        self.assertEqual(result[0]['retrieval']['reference_answer'], 'wrong')
        result[0]['native_output_ids_by_repetition'] = [[1, 0, 9], [9, 0, 8]]
        compare.check_answers(cases, result, lambda ids: ' 1234\n' if ids == [1] else 'wrong', 0)
        self.assertEqual(result[0]['retrieval']['native_answers'], ['1234', 'wrong'])
        self.assertFalse(result[0]['retrieval']['native_correct'])


if __name__ == '__main__':
    unittest.main()
