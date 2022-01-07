"""Unit tests for the main detector code"""

import pytest
from copydetect import CopyDetector, CodeFingerprint, compare_files, utils
import numpy as np
from pathlib import Path

TESTS_DIR = str(Path(__file__).parent)

class TestTwoFileDetection():
    """Test of the user-facing copydetect code for a simple two-file
    case. The two files both use several sections from a boilerplate
    file but are otherwise different.
    """
    def test_compare(self):
        config = {
          "test_directories" : [TESTS_DIR + "/sample_py/code"],
          "reference_directories" : [TESTS_DIR + "/sample_py/code"],
          "extensions" : ["py"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 25,
          "display_threshold" : 0
        }
        detector = CopyDetector(config, silent=True)
        detector.run()

        assert np.array_equal(np.array([[-1,1137/2052],[1137/1257,-1]]),
                              detector.similarity_matrix)
        assert np.array_equal(np.array([[-1,1137],[1137,-1]]),
                              detector.token_overlap_matrix)

        html_out = detector.generate_html_report(output_mode="return")

        # verify highlighting code isn't being escaped
        test_str1 = "data[2] = [</span>0<span class='highlight-red'>, 6, 1]"
        test_str2 = "data[2] = [</span>3<span class='highlight-green'>, 6, 1]"
        # verify input code is being escaped
        test_str3 = "print(&#34;Incorrect num&#34;"
        assert test_str1 in html_out
        assert test_str2 in html_out
        assert test_str3 in html_out

    def test_compare_manual_config(self):
        detector = CopyDetector(noise_t=25, guarantee_t=25, silent=True)
        detector.add_file(TESTS_DIR + "/sample_py/code/sample1.py")
        detector.add_file(TESTS_DIR + "/sample_py/code/sample2.py")
        detector.run()

        assert np.array_equal(np.array([[-1,1137/2052],[1137/1257,-1]]),
                              detector.similarity_matrix)
        assert np.array_equal(np.array([[-1,1137],[1137,-1]]),
                              detector.token_overlap_matrix)

    def test_compare_saving(self, tmpdir):
        config = {
          "test_directories" : [TESTS_DIR + "/sample_py/code"],
          "reference_directories" : [TESTS_DIR + "/sample_py/code"],
          "extensions" : ["py"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 25,
          "display_threshold" : 0,
          "disable_autoopen" : True,
          "out_file" : tmpdir
        }
        detector = CopyDetector(config, silent=True)
        detector.run()
        detector.generate_html_report()

        # check for expected files
        assert Path(tmpdir + "/report.html").exists()

    def test_compare_boilerplate(self):
        config = {
          "test_directories" : [TESTS_DIR + "/sample_py/code"],
          "reference_directories" : [TESTS_DIR + "/sample_py/code"],
          "boilerplate_directories" : [TESTS_DIR + "/sample_py/boilerplate"],
          "extensions" : ["py"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 25,
          "display_threshold" : 0
        }
        detector = CopyDetector(config, silent=True)
        detector.run()

        assert np.array_equal(np.array([[-1,0],[0,-1]]),
                              detector.similarity_matrix)
        assert np.array_equal(np.array([[-1,0],[0,-1]]),
                              detector.token_overlap_matrix)

    def test_severalfiles(self, tmpdir):
        """Run the detector over all the files in the tests directory
        and perform some basic sanity checking.
        """
        config = {
          "test_directories" : [TESTS_DIR],
          "reference_directories" : [TESTS_DIR],
          "extensions" : ["*"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 30,
          "display_threshold" : 0.3,
          "disable_autoopen" : True,
          "out_file" : tmpdir
        }
        detector = CopyDetector(config, silent=True)
        detector.run()
        html_out = detector.generate_html_report()

        skipped_files = detector.similarity_matrix == -1
        assert np.all(detector.similarity_matrix[~skipped_files] >= 0)
        assert np.any(detector.similarity_matrix[~skipped_files] > 0)
        assert np.all(detector.similarity_matrix[~skipped_files] <= 1)
        assert np.all(detector.token_overlap_matrix[~skipped_files] >= 0)

class TestTwoFileAPIDetection():
    """Performs the same checks as the other two-file check, but uses
    the API instead of the command line code.
    """
    def test_compare(self):
        fp1 = CodeFingerprint(25, 1, file=TESTS_DIR+"/sample_py/code/sample1.py")
        fp2 = CodeFingerprint(25, 1, file=TESTS_DIR+"/sample_py/code/sample2.py")
        token_overlap, similarities, slices = compare_files(fp1, fp2)

        assert token_overlap == 1137
        assert similarities[0] == 1137/2052
        assert similarities[1] == 1137/1257

    def test_code_str(self):
        sample1 = """import numpy as np
        import matplotlib.pyplot as plt
        import sys

        def generate_training_data_binary(num):
        if num == 1:
            data = np.zeros((10,3))
            for i in range(5):
            data[i] = [i-5, 0, 1]
            data[i+5] = [i+1, 0, -1]

        elif num == 2:
            data = np.zeros((10,3))
            for i in range(5):
            data[i] = [0, i-5, 1]
            data[i+5] = [0, i+1, -1]

        elif num == 3:
            data = np.zeros((10,3))
            data[0] = [3, 2, 1]
            data[1] = [6, 2, 1]
            data[2] = [0, 6, 1]
            data[3] = [4, 4, 1]
            data[4] = [5, 4, 1]
            data[5] = [-1, -2, -1]
            data[6] = [-2, -4, -1]
            data[7] = [-3, -3, -1]
            data[8] = [-4, -2, -1]
            data[9] = [-4, -4, -1]
        elif num == 4:
            data = np.zeros((10,3))
            data[0] = [-1, 1, 1]
            data[1] = [-2, 2, 1]
            data[2] = [-3, 5, 1]
            data[3] = [-3, -1, 1]
            data[4] = [-2, 1, 1]
            data[5] = [3, -6, -1]
            data[6] = [0, -2, -1]
            data[7] = [-1, -7, -1]
            data[8] = [1, -10, -1]
            data[9] = [0, -8, -1]

        else:
            print("Incorrect num", num, "provided to generate_training_data_binary.")
            sys.exit()

        return data

        def generate_training_data_multi(num):
        if num == 1:
            data = np.zeros((20,3))
            for i in range(5):
            data[i] = [i-5, 0, 1]
            data[i+5] = [i+1, 0, 2]
            data[i+10] = [0, i-5, 3]
            data[i+15] = [0, i+1, 4]
            Y = 4

        elif num == 2:
            data = np.zeros((15,3))
            data[0] = [-5, -5, 1]
            data[1] = [-3, -2, 1]
            data[2] = [-5, -3, 1]
            data[3] = [-5, -4, 1]
            data[4] = [-2, -9, 1]
            data[5] = [0, 6, 2]
            data[6] = [-1, 3, 2]
            data[7] = [-2, 1, 2]
            data[8] = [1, 7, 2]
            data[9] = [1, 5, 2]
            data[10] = [6, 3, 3]
            data[11] = [9, 2, 3]
            data[12] = [10, 4, 3]
            data[13] = [8, 1, 3]
            data[14] = [9, 0, 3]
            Y = 3

        else:
            print("Incorrect num", num, "provided to generate_training_data_binary.")
            sys.exit()

        return [data, Y]

        def plot_training_data_binary(data, boundary=None):
        for item in data:
            if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+')
            else:
            plt.plot(item[0], item[1], 'ro')
        m = max(data.max(), abs(data.min()))+1
        plt.axis([-m, m, -m, m])

        if boundary:
            w = boundary[0]
            b = boundary[1]
            s = boundary[2]

            if (w[0] == 0):
                x2_min = -(-m*w[0]+b)/(w[1])
                x2_max = -( m*w[0]+b)/(w[1])

                diff = -(s[0][0]*w[0]+b)/(w[1]) - s[0][1]
                plt.plot((-m, m), (x2_min+diff, x2_max+diff), color='black',ls='dashed')
                plt.plot((-m, m), (x2_min-diff, x2_max-diff), color='black',ls='dashed')
                plt.plot((-m, m), (x2_min, x2_max), color='black')
            else:
                x1_min = -(-m*w[1]+b)/(w[0])
                x1_max = -( m*w[1]+b)/(w[0])

                diff = -(s[0][1]*w[1]+b)/(w[0]) - s[0][0]
                plt.plot((x1_min+diff, x1_max+diff), (-m, m), color='black',ls='dashed')
                plt.plot((x1_min-diff, x1_max-diff), (-m, m), color='black',ls='dashed')
                plt.plot((x1_min, x1_max), (-m, m), color='black')

            for sv in s:
                plt.scatter(sv[0], sv[1], s=200, facecolors='none', edgecolors='black')
        plt.show()

        def plot_training_data_multi(data, boundaries=None, plot_margin=False):
        colors = ['b', 'r', 'g', 'm']
        shapes = ['+', 'o', '*', '.']

        for item in data:
            plt.plot(item[0], item[1], colors[int(item[2])-1] + shapes[int(item[2])-1])
        m = max(data.max(), abs(data.min()))+1
        plt.axis([-m, m, -m, m])

        if boundaries:
            for i in range(len(boundaries[0])):
            w = boundaries[0][i]
            b = boundaries[1][i]
            s = boundaries[2][i]

            if (w[0] == 0):
                x2_min = -(-m*w[0]+b)/(w[1])
                x2_max = -( m*w[0]+b)/(w[1])

                if plot_margin:
                    diff = -(s[0][0]*w[0]+b)/(w[1]) - s[0][1]
                    plt.plot((-m, m), (x2_min+diff, x2_max+diff), color='black',ls='dashed')
                    plt.plot((-m, m), (x2_min-diff, x2_max-diff), color='black',ls='dashed')
                plt.plot((-m, m), (x2_min, x2_max), color='black')
            else:
                x1_min = -(-m*w[1]+b)/(w[0])
                x1_max = -( m*w[1]+b)/(w[0])

                if plot_margin:
                    diff = -(s[0][1]*w[1]+b)/(w[0]) - s[0][0]
                    plt.plot((x1_min+diff, x1_max+diff), (-m, m), color='black',ls='dashed')
                    plt.plot((x1_min-diff, x1_max-diff), (-m, m), color='black',ls='dashed')
                plt.plot((x1_min, x1_max), (-m, m), color='black')
            if plot_margin:
                for sv in s:
                    plt.scatter(sv[0], sv[1], s=200, facecolors='none', edgecolors='black')

        plt.show()"""
        sample2 = """import numpy as np
        import matplotlib.pyplot as plt
        import sys
        import svm

        def generate_training_data_binary(num):
        if num == 1:
            data = np.zeros((10,3))
            for i in range(5):
            data[i] = [i-5, 0, 1]
            data[i+5] = [i+1, 0, -1]
            
        elif num == 2:
            data = np.zeros((10,3))
            for i in range(5):
            data[i] = [0, i-5, 1]
            data[i+5] = [0, i+1, -1]

        elif num == 3:
            data = np.zeros((10,3))
            data[0] = [3, 2, 1]
            data[1] = [6, 2, 1]
            data[2] = [3, 6, 1]
            data[3] = [4, 4, 1]
            data[4] = [5, 4, 1]
            data[5] = [-1, -2, -1]
            data[6] = [-2, -4, -1]
            data[7] = [-3, -3, -1]
            data[8] = [-4, -2, -1]
            data[9] = [-4, -4, -1]
        elif num == 4:
            data = np.zeros((10,3))
            data[0] = [-1, 1, 1]
            data[1] = [-2, 2, 1]
            data[2] = [-3, 5, 1]
            data[3] = [-3, -1, 1]
            data[4] = [-2, 1, 1]
            
            data[5] = [3, -6, -1]
            data[6] = [0, -2, -1]
            data[7] = [-1, -7, -1]
            data[8] = [1, -10, -1]
            data[9] = [0, -8, -1]

        else:
            print("Incorrect num", num, "provided to generate_training_data_binary.")
            sys.exit()

        return data

        data = generate_training_data_binary(4)
        w, b, s = svm.svm_train_brute(data)
        print(w)
        print(b)
        pt= np.array([1,3])
        dist = svm.distance_point_to_hyperplane(pt, w, b)
        #print (dist)
        margin = svm.compute_margin(data, w, b)




        def generate_training_data_multi(num):
        if num == 1:
            data = np.zeros((20,3))
            for i in range(5):
            data[i] = [i-5, 0, 1]
            data[i+5] = [i+1, 0, 2]
            data[i+10] = [0, i-5, 3]
            data[i+15] = [0, i+1, 4]
            Y = 4

        elif num == 2:
            data = np.zeros((15,3))
            data[0] = [-5, -5, 1]
            data[1] = [-3, -2, 1]
            data[2] = [-5, -3, 1]
            data[3] = [-5, -4, 1]
            data[4] = [-2, -9, 1]
            data[5] = [0, 6, 2]
            data[6] = [-1, 3, 2]
            data[7] = [-2, 1, 2]
            data[8] = [1, 7, 2]
            data[9] = [1, 5, 2]
            data[10] = [6, 3, 3]
            data[11] = [9, 2, 3]
            data[12] = [10, 4, 3]
            data[13] = [8, 1, 3]
            data[14] = [9, 0, 3]
            Y = 3

        else:
            print("Incorrect num", num, "provided to generate_training_data_binary.")
            sys.exit()

        return [data, Y]

        def plot_training_data_binary(data):
        for item in data:
            if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+')
            else:
            plt.plot(item[0], item[1], 'ro')
        m = max(data.max(), abs(data.min()))+1

        plt.axis([-m, m, -m, m])
        plt.show()

        def plot_training_data_multi(data):
        colors = ['b', 'r', 'g', 'm']
        shapes = ['+', 'o', '*', '.']

        for item in data:
            plt.plot(item[0], item[1], colors[int(item[2])-1] + shapes[int(item[2])-1])
        m = max(data.max(), abs(data.min()))+1
        plt.axis([-m, m, -m, m])
        plt.show()

        plot = plot_training_data_binary(data)"""
        fp1 = CodeFingerprint(25, 1, code=sample1, language='python')
        fp2 = CodeFingerprint(25, 1, code=sample2, language='python')
        token_overlap, similarities, slices = compare_files(fp1, fp2)

        code1, _ = utils.highlight_overlap(fp1.raw_code, slices[0], ">>", "<<")
        code2, _ = utils.highlight_overlap(fp2.raw_code, slices[1], ">>", "<<")
        
        assert token_overlap == 1137
        assert similarities[0] == 1137/2052
        assert similarities[1] == 1137/1257

    def test_compare_boilerplate(self):
        bp_fingerprint = CodeFingerprint(25, 1, file=TESTS_DIR+"/sample_py/boilerplate/handout.py")
        fp1 = CodeFingerprint(25, 1, file=TESTS_DIR+"/sample_py/code/sample1.py", boilerplate=bp_fingerprint.hashes)
        fp2 = CodeFingerprint(25, 1, file=TESTS_DIR+"/sample_py/code/sample2.py", boilerplate=bp_fingerprint.hashes)

        token_overlap, similarities, slices = compare_files(fp1, fp2)

        assert token_overlap == 0
        assert similarities[0] == 0
        assert similarities[1] == 0

class TestParameters():
    """Test cases for individual parameters"""
    def test_ignore_leaf(self):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
                                ignore_leaf=True, silent=True)
        detector.run()

        # sample1 and sample2 should not have been compared
        # + 4 self compares = 6 total skips
        assert np.sum(detector.similarity_matrix == -1) == 6

    def test_same_name_only(self):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
                                same_name_only=True, silent=True)
        detector.run()

        # the only comparison should be between the two handout.py files
        assert np.sum(detector.similarity_matrix != -1) == 2

    def test_disable_filtering(self):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
                                disable_filtering=True, silent=True)
        detector.run()

        fingerprint1 = detector.file_data[
            str(Path(TESTS_DIR + "/sample_py/code/sample1.py"))]
        assert fingerprint1.raw_code == fingerprint1.filtered_code

    def test_force_language(self):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
                                force_language="java", silent=True)
        detector.run()

        fingerprint1 = detector.file_data[
            str(Path(TESTS_DIR + "/sample_py/handout.py"))]

        # "#" isn't a comment in java, so it won't be removed
        assert fingerprint1.filtered_code[0] == "#"

    def test_truncation(self):
        detector = CopyDetector(
            test_dirs=[TESTS_DIR + "/sample_py/boilerplate"],
            noise_t=10, guarantee_t=10, truncate=True, silent=True)
        detector.add_file(str(Path(TESTS_DIR + "/sample_py/handout.py")))
        detector.run()
        code_list = detector.get_copied_code_list()

        assert len(code_list[0][4]) < 500 and len(code_list[0][5]) < 500

    def test_out_file(self, tmpdir):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
            silent=True, out_file=tmpdir + "/test", autoopen=False)
        detector.run()
        detector.generate_html_report()

        assert Path(tmpdir + "/test.html").exists()

        with pytest.raises(ValueError):
            detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
            silent=True, out_file=tmpdir + "/not_a_dir/test")

        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
            silent=True, out_file=tmpdir, autoopen=False)
        detector.run()
        detector.generate_html_report()

        assert Path(tmpdir + "/report.html").exists()
