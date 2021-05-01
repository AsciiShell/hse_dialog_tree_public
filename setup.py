from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def load_requirements(filename: str) -> list:
    requirements = []
    with open(filename, 'r') as f:
        for requirement in parse_requirements(f.read()):
            extras = '[{}]'.format(','.join(requirement.extras)) if requirement.extras else ''
            requirements.append(
                '{}{}{}'.format(requirement.name, extras, requirement.specifier)
            )
    return requirements


module_name = 'hse_dialog_tree'

with open('README.md', 'rt') as f:
    long_description = f.read()

setup(
    name=module_name,
    version='0.2.3',
    description='Выпускная квалификационная работа',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AsciiShell/hse_dialog_tree',
    author='asciishell (Aleksey Podchezertsev)',
    author_email='dev@asciishell.ru',
    license='copyright',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
    ],
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            '{0} = {0}.__main__:main'.format(module_name),
            '{0}_download = {0}.utils.data.download_tools:main'.format(module_name),
        ]
    },
    include_package_data=True,
    zip_safe=False,
    # install_requires=load_requirements('requirements.txt'),
)
