"""Unit tests for the main detector code"""

import logging
from pathlib import Path

import numpy as np
import pytest

from copydetect import CodeFingerprint, CopyDetector, compare_files, utils

TESTS_DIR = str(Path(__file__).parent)


class TestTwoFileDetection:
    """Test of the user-facing copydetect code for a simple two-file
    case. The two files both use several sections from a boilerplate
    file but are otherwise different.
    """

    def test_compare(self):
        config = {
            "test_directories": [TESTS_DIR + "/sample_py/code"],
            "reference_directories": [TESTS_DIR + "/sample_py/code"],
            "extensions": ["py"],
            "noise_threshold": 25,
            "guarantee_threshold": 25,
            "display_threshold": 0,
        }
        detector = CopyDetector(config, silent=True)
        detector.run()

        assert np.array_equal(
            np.array([[-1, 1138 / 2058], [1138 / 1258, -1]]),
            detector.similarity_matrix,
        )
        assert np.array_equal(
            np.array([[-1, 1138], [1138, -1]]), detector.token_overlap_matrix
        )

        html_out = detector.generate_html_report(output_mode="return")
        logging.error(html_out)
        # verify highlighting code isn't being escaped
        test_str1 = "data[2] = [</span>0<span class='highlight-red'>, 6, 1]"
        test_str2 = "data[2] = [</span>3<span class='highlight-green'>, 6, 1]"
        # verify input code is being escaped
        test_str3 = "&#34;Incorrect num&#34;"
        assert test_str1 in html_out
        assert test_str2 in html_out
        assert test_str3 in html_out

    def test_compare_manual_config(self):
        detector = CopyDetector(noise_t=25, guarantee_t=25, silent=True)
        detector.add_file(TESTS_DIR + "/sample_py/code/sample1.py")
        detector.add_file(TESTS_DIR + "/sample_py/code/sample2.py")
        detector.run()

        assert np.array_equal(
            np.array([[-1, 1138 / 2058], [1138 / 1258, -1]]),
            detector.similarity_matrix,
        )
        assert np.array_equal(
            np.array([[-1, 1138], [1138, -1]]), detector.token_overlap_matrix
        )

    def test_compare_saving(self, tmpdir):
        config = {
            "test_directories": [TESTS_DIR + "/sample_py/code"],
            "reference_directories": [TESTS_DIR + "/sample_py/code"],
            "extensions": ["py"],
            "noise_threshold": 25,
            "guarantee_threshold": 25,
            "display_threshold": 0,
            "disable_autoopen": True,
            "out_file": tmpdir,
        }
        detector = CopyDetector(config, silent=True)
        detector.run()
        detector.generate_html_report()

        # check for expected files
        assert Path(tmpdir + "/report.html").exists()

    def test_compare_boilerplate(self):
        config = {
            "test_directories": [TESTS_DIR + "/sample_py/code"],
            "reference_directories": [TESTS_DIR + "/sample_py/code"],
            "boilerplate_directories": [TESTS_DIR + "/sample_py/boilerplate"],
            "extensions": ["py"],
            "noise_threshold": 25,
            "guarantee_threshold": 25,
            "display_threshold": 0,
        }
        detector = CopyDetector(config, silent=True)
        detector.run()

        assert np.array_equal(
            np.array([[-1, 0], [0, -1]]), detector.similarity_matrix
        )
        assert np.array_equal(
            np.array([[-1, 0], [0, -1]]), detector.token_overlap_matrix
        )

    def test_severalfiles(self, tmpdir):
        """Run the detector over all the files in the tests directory
        and perform some basic sanity checking.
        """
        config = {
            "test_directories": [TESTS_DIR],
            "reference_directories": [TESTS_DIR],
            "extensions": ["*"],
            "noise_threshold": 25,
            "guarantee_threshold": 30,
            "display_threshold": 0.3,
            "disable_autoopen": True,
            "out_file": tmpdir,
        }
        detector = CopyDetector(config, silent=True)
        detector.run()

        skipped_files = detector.similarity_matrix == -1
        assert np.all(detector.similarity_matrix[~skipped_files] >= 0)
        assert np.any(detector.similarity_matrix[~skipped_files] > 0)
        assert np.all(detector.similarity_matrix[~skipped_files] <= 1)
        assert np.all(detector.token_overlap_matrix[~skipped_files] >= 0)


class TestTwoFileAPIDetection:
    """Performs the same checks as the other two-file check, but uses
    the API instead of the command line code.
    """

    def test_compare(self):
        fp1 = CodeFingerprint(
            25, 1, file=TESTS_DIR + "/sample_py/code/sample1.py"
        )
        fp2 = CodeFingerprint(
            25, 1, file=TESTS_DIR + "/sample_py/code/sample2.py"
        )
        token_overlap, similarities, slices = compare_files(fp1, fp2)

        assert token_overlap == 1138
        assert similarities[0] == 1138 / 2058
        assert similarities[1] == 1138 / 1258

    def test_code_str(self):
        file1 = TESTS_DIR + "/sample_py/code/sample1.py"
        with open(file1, errors="ignore") as code_fp:
            sample1 = code_fp.read()

        file2 = TESTS_DIR + "/sample_py/code/sample2.py"
        with open(file2, errors="ignore") as code_fp:
            sample2 = code_fp.read()
        fp1 = CodeFingerprint(25, 1, code=sample1, language="python")
        fp2 = CodeFingerprint(25, 1, code=sample2, language="python")
        token_overlap, similarities, slices = compare_files(fp1, fp2)

        code1, _ = utils.highlight_overlap(fp1.raw_code, slices[0], ">>", "<<")
        code2, _ = utils.highlight_overlap(fp2.raw_code, slices[1], ">>", "<<")
        assert token_overlap == 1138
        assert similarities[0] == 1138 / 2058
        assert similarities[1] == 1138 / 1258

    def test_compare_boilerplate(self):
        bp_fingerprint = CodeFingerprint(
            25, 1, file=TESTS_DIR + "/sample_py/boilerplate/handout.py"
        )
        fp1 = CodeFingerprint(
            25,
            1,
            file=TESTS_DIR + "/sample_py/code/sample1.py",
            boilerplate=bp_fingerprint.hashes,
        )
        fp2 = CodeFingerprint(
            25,
            1,
            file=TESTS_DIR + "/sample_py/code/sample2.py",
            boilerplate=bp_fingerprint.hashes,
        )

        token_overlap, similarities, slices = compare_files(fp1, fp2)

        assert token_overlap == 0
        assert similarities[0] == 0
        assert similarities[1] == 0


class TestParameters:
    """Test cases for individual parameters"""

    def test_ignore_leaf(self):
        detector = CopyDetector(
            test_dirs=[TESTS_DIR + "/sample_py"], ignore_leaf=True, silent=True
        )
        detector.run()

        # sample1 and sample2 should not have been compared
        # + 4 self compares = 6 total skips
        assert np.sum(detector.similarity_matrix == -1) == 6

    def test_same_name_only(self):
        detector = CopyDetector(
            test_dirs=[TESTS_DIR + "/sample_py"],
            same_name_only=True,
            silent=True,
        )
        detector.run()

        # the only comparison should be between the two handout.py files
        assert np.sum(detector.similarity_matrix != -1) == 2

    def test_disable_filtering(self):
        detector = CopyDetector(
            test_dirs=[TESTS_DIR + "/sample_py"],
            disable_filtering=True,
            silent=True,
        )
        detector.run()

        fingerprint1 = detector.file_data[
            str(Path(TESTS_DIR + "/sample_py/code/sample1.py"))
        ]
        assert fingerprint1.raw_code == fingerprint1.filtered_code

    def test_force_language(self):
        detector = CopyDetector(
            test_dirs=[TESTS_DIR + "/sample_py"],
            force_language="java",
            silent=True,
        )
        detector.run()

        fingerprint1 = detector.file_data[
            str(Path(TESTS_DIR + "/sample_py/handout.py"))
        ]

        # "#" isn't a comment in java, so it won't be removed
        assert fingerprint1.filtered_code[0] == "#"

    def test_truncation(self):
        detector = CopyDetector(
            test_dirs=[TESTS_DIR + "/sample_py/boilerplate"],
            noise_t=10,
            guarantee_t=10,
            truncate=True,
            silent=True,
        )
        detector.add_file(str(Path(TESTS_DIR + "/sample_py/handout.py")))
        detector.run()
        code_list = detector.get_copied_code_list()

        assert len(code_list[0][4]) < 500 and len(code_list[0][5]) < 500

    def test_out_file(self, tmpdir):
        detector = CopyDetector(
            test_dirs=[TESTS_DIR + "/sample_py"],
            silent=True,
            out_file=tmpdir + "/test",
            autoopen=False,
        )
        detector.run()
        detector.generate_html_report()

        assert Path(tmpdir + "/test.html").exists()

        with pytest.raises(ValueError):
            detector = CopyDetector(
                test_dirs=[TESTS_DIR + "/sample_py"],
                silent=True,
                out_file=tmpdir + "/not_a_dir/test",
            )

        detector = CopyDetector(
            test_dirs=[TESTS_DIR + "/sample_py"],
            silent=True,
            out_file=tmpdir,
            autoopen=False,
        )
        detector.run()
        detector.generate_html_report()

        assert Path(tmpdir + "/report.html").exists()
