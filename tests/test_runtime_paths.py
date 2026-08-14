"""运行配置路径与仓库隔离测试。"""

import unittest

from wd_subtitler.runtime_paths import APP_DIR, CONFIG_FILE, USER_DATA_DIR


class RuntimePathsTests(unittest.TestCase):
    def test_API配置位于仓库之外(self):
        self.assertEqual(USER_DATA_DIR / "config.json", CONFIG_FILE)
        self.assertNotEqual(APP_DIR / "config.json", CONFIG_FILE)
        self.assertFalse(CONFIG_FILE.is_relative_to(APP_DIR))


if __name__ == "__main__":
    unittest.main()
