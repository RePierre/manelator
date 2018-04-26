from setuptools import setup, find_packages

version = '0.1'

install_requires = [
    'tensorflow'
]

dev_requires = [
    "autopep8",
    "rope_py3k",
    "jedi",
    "importmagic",
    "yapf"
]

tests_requires = [
]

setup(
    name='manelator',
    version=version,
    description="Manele generator using LSTM.",
    long_description="",
    classifiers=[],
    keywords="",
    author="RePierre",
    author_email="",
    url="",
    license="",
    packages=find_packages(exclude=['']),
    include_package_data=True,
    zip_safe=True,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require={
        'dev': dev_requires
    },
    test_suite="py.test",
)
