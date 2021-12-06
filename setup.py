#setup.py based on https://www.benjack.io/2018/02/02/python-cpp-revisited.html by Benjamin R. Jack and https://github.com/sizmailov/pyxmolpp2/blob/master/setup.py by Sergei Izmailov
import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < LooseVersion('3.5.0'):
            raise RuntimeError("CMake >= 3.5.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        build_type = os.environ.get("BUILD_TYPE", "Release")
        build_args = ['--config', build_type]

        # Pile all .so in one place and use $ORIGIN as RPATH
        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        cmake_args += ["-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE"] 
        cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("$ORIGIN")]
        cmake_args += ["-DPYBIND11_VERSION_INFO="+self.distribution.get_version()]
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(build_type.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + build_type]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake',
                               '--build', '.',
                               '--target', os.path.basename(ext.name)
                               ] + build_args,
                              cwd=self.build_temp)

setup(
    name='naphash_py',
    version='0.8',
    author='Oliver Zendel',
    author_email='oliver.zendel@ait.ac.at',
    description='Calculate NAPHash/NPHash image hashes',
    long_description=open("README.md").read(),
    ext_modules=[CMakeExtension('naphash_py')],
    include_package_data=True,
    test_suite='tests',
    packages=find_packages('src'),
    package_dir={'':'src'},
    cmdclass=dict(build_ext=CMakeBuild),
    url="https://github.com/ozendelait/naphash",
    zip_safe=False,
    install_requires=[
        "numpy",
        "pybind11"
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ],
)
