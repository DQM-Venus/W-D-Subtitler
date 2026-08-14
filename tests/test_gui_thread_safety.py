"""GUI 后台处理函数的线程安全约束测试。"""

import ast
import inspect
import textwrap
import unittest

from wd_subtitler.gui import SubtitleToolApp
from wd_subtitler.processing_pipeline import SubtitleProcessingPipeline


class GUIThreadSafetyTests(unittest.TestCase):
    def test_后台流程不直接访问Tk控件(self):
        forbidden_prefixes = ("entry_", "combo_", "var_", "btn_", "lbl_")
        forbidden_names = {"root", "media_tree", "progress", "log_area"}
        methods = [
            SubtitleProcessingPipeline.run,
            SubtitleProcessingPipeline._run_internal,
            SubtitleProcessingPipeline._run_primary_asr,
            SubtitleProcessingPipeline._run_large_v3_review,
            SubtitleProcessingPipeline._translate_and_save,
            SubtitleToolApp._run_pipeline,
        ]

        for method in methods:
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
            names = {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            }
            forbidden = {
                name for name in names
                if name.startswith(forbidden_prefixes) or name in forbidden_names
            }
            self.assertEqual(set(), forbidden, method.__name__)


if __name__ == "__main__":
    unittest.main()
