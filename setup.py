import setuptools

setuptools.setup(
    name='gym_utils',
    version='0.1',
    description='Open AI gym utilities.',
    install_requires=[
        'gym',
        'numpy',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
    test_suite='pytest', )
