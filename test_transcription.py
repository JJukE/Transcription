import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import transcription_backends as backend
import run_container as runner
from node_resource_gate import assess_node_resources, add_gpu_evidence


class TextTests(unittest.TestCase):
    def test_korean_punctuation_and_spaces_survive_alignment(self):
        text = '안녕하세요, Qwen 모델입니다!'
        words = backend.aligned_words(text, [
            {'text': '안녕하세요', 'start': 0, 'end': 1},
            {'text': 'Qwen', 'start': 1, 'end': 2},
            {'text': '모델입니다', 'start': 2, 'end': 3}])
        self.assertEqual(''.join(w['word'] for w in words), text)
        self.assertEqual(''.join(s['text'] for s in backend.word_segments(words)), text)

    def test_mismatched_alignment_fails(self):
        with self.assertRaises(ValueError):
            backend.aligned_words('앞 단어', [{'text': '단어', 'start': 1, 'end': 2}])

    def test_overlap_midpoint_is_owned_by_one_chunk(self):
        words = [{'word': '경계', 'start': 1, 'end': 3}]
        left = backend.crop_words(words, 118, 0, 120, False, 240)
        right = backend.crop_words(words, 118, 120, 240, True, 240)
        self.assertEqual(left, [])
        self.assertEqual(len(right), 1)
        self.assertEqual(right[0]['start'], 119)

    def test_out_of_audio_timestamp_fails(self):
        with self.assertRaises(ValueError):
            backend.crop_words([{'word': 'x', 'start': -1, 'end': 1}], 0, 0, 2, True, 2)

    def test_languages(self):
        self.assertEqual(backend.qwen_language('ko'), 'Korean')
        self.assertEqual(backend.qwen_language('Korean'), 'Korean')
        self.assertIsNone(backend.qwen_language('auto'))

    def test_all_outputs_do_not_invent_confidence(self):
        result = {'text': '테스트!', 'language': 'Korean', 'segments': [
            {'start': 0.125, 'end': 2.5, 'text': '테스트!', 'words': []}]}
        with tempfile.TemporaryDirectory() as directory:
            backend.write_qwen(result, directory, 'result')
            self.assertEqual(len(list(Path(directory).iterdir())), 5)
            saved = json.loads(Path(directory, 'result.json').read_text())
            self.assertEqual(saved, result)
            self.assertNotIn('confidence', saved['segments'][0])
            self.assertIn('00:00:00,125 --> 00:00:02,500', Path(directory,'result.srt').read_text())
            self.assertTrue(Path(directory, 'result.vtt').read_text().startswith('WEBVTT'))
            self.assertIn('125\t2500\t테스트!', Path(directory, 'result.tsv').read_text())


class NetworkTests(unittest.TestCase):
    def test_isolation_rejects_leftover_interface(self):
        with patch.object(runner, 'inspect', return_value={'NetworkSettings': {'Networks': {}}}), patch.object(runner, 'docker', side_effect=['lo\neth0', 'header', '']):
            with self.assertRaises(RuntimeError):
                runner.verify_isolated('test')

    def test_isolation_rejects_leftover_endpoint(self):
        with patch.object(runner, 'inspect', return_value={'NetworkSettings': {'Networks': {'bridge': {}}}}):
            with self.assertRaises(RuntimeError):
                runner.verify_isolated('test')

    def test_restore_never_reconnects_a_running_task(self):
        with patch.object(runner, 'task_finished', return_value=False), patch.object(runner, 'docker') as docker:
            with self.assertRaises(RuntimeError):
                runner.restore('test', {'task_started': True, 'token': 'a'})
            docker.assert_not_called()

    def test_unknown_or_busy_resource_gate_fails_closed(self):
        self.assertEqual(assess_node_resources(None, 1)['decision'], 'unknown')
        self.assertEqual(assess_node_resources(85, 1)['decision'], 'blocked')
        gate = add_gpu_evidence(assess_node_resources(1,1), [{'index':0,'compute_processes':[{'pid':123}]}], selected_gpu=0, gpu_required=True)
        self.assertEqual(gate['decision'], 'blocked')


if __name__ == '__main__':
    unittest.main()
