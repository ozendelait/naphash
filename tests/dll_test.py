import unittest
import numpy as np
import os, shutil

#find directory housing python module
def return_which(modul_name, check_lnk = True):
  import sys, glob
  for p in sys.path:
    test_p = os.path.join(os.path.realpath(p),modul_name+'.pyd')
    if os.path.exists(test_p):
      return test_p
    if not check_lnk:
      continue
    test_lnk = os.path.join(os.path.realpath(p),modul_name+'.egg-link')
    if not os.path.exists(test_lnk):
      continue
    print("checking",test_lnk)
    with open(test_lnk) as ifile:
      check_dir = ifile.readline()
    if len(check_dir) < 5:
      continue
    test_p = os.path.join(os.path.realpath(check_dir),modul_name+'.pyd')
    if os.path.exists(test_p):
      return test_p
  return None

#find dll within installed OpenCV_DIR
def find_ocv_dll(root_dir, comp):
  import os, glob, itertools
  print("CHECKING",root_dir)
  res = itertools.chain.from_iterable(glob.iglob(os.path.join(root,comp+'*.dll')) for root, dirs, files in os.walk(root_dir))
  found_dlls = [r for r in res]
  return found_dlls[0]

class MainTest(unittest.TestCase):
    def test_dll(self):
        try:
          from naphash_py import naphash as nhpy, rot_inv_type
        except ImportError as e:
          trg_dir = os.path.dirname(return_which("naphash_py"))
          ocv_dir = os.environ[[k for k in os.environ if 'opencv_dir' in k.lower()][0]]
          for comp in ['opencv_core', 'opencv_imgproc']:
            src_file = find_ocv_dll(ocv_dir, comp)
            shutil.copy2(src_file,trg_dir)
        import naphash_py
        self.assertEqual(naphash_py.__name__,"naphash_py")

if __name__ == '__main__':
    unittest.main()